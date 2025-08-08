#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked model with data engineering + hyperparameter search.
Outputs a Kaggle-style submission CSV.

Usage example:
python stack_search_train.py \
  --train ./data/training.csv \
  --test ./data/test.csv \
  --sample ./data/sample_submission.csv \
  --out submission.csv \
  --trials 40 \
  --seeds 42 777 \
  --folds 5
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Tree models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def log(msg: str):
    print(msg, flush=True)

def safe_numeric_cols(df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    """Return numeric feature columns excluding drops."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in drop_cols]

def add_interactions(df: pd.DataFrame, base_cols: List[str], max_pairs: int = 60) -> List[str]:
    """Create simple pairwise interactions: sum, diff, prod, ratio (safe).
       Limit count by choosing top variance columns."""
    # pick top by variance
    var_rank = df[base_cols].var().sort_values(ascending=False).index.tolist()
    top = var_rank[:min(len(var_rank), max(8, int(len(var_rank) * 0.5)))]
    pairs = []
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            pairs.append((top[i], top[j]))
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    new_cols = []
    for a,b in pairs:
        c1 = f"{a}__plus__{b}"
        c2 = f"{a}__minus__{b}"
        c3 = f"{a}__times__{b}"
        c4 = f"{a}__ratio__{b}"
        df[c1] = df[a] + df[b]
        df[c2] = df[a] - df[b]
        df[c3] = df[a] * df[b]
        # safe ratio
        eps = 1e-6 * (df[b].abs().mean() + 1)
        df[c4] = df[a] / (df[b].abs() + eps)
        new_cols += [c1,c2,c3,c4]
    return new_cols

def add_per_query_features(df: pd.DataFrame, group_col: str, feat_cols: List[str]) -> List[str]:
    """Add per-query ranks and z-scores (NO target leakage; uses only X)."""
    new_cols = []
    g = df.groupby(group_col, observed=True)

    # rank (dense) within query
    for c in feat_cols:
        rname = f"{c}__qrank"
        df[rname] = g[c].rank(method="first", ascending=True).astype(np.float32)
        new_cols.append(rname)

    # z-score within query
    stats = g[feat_cols].agg(['mean','std'])
    # map mean/std
    for c in feat_cols:
        m = stats[(c, 'mean')].reindex(df[group_col]).values
        s = stats[(c, 'std')].reindex(df[group_col]).values
        s = np.where(s==0, 1.0, s)
        zname = f"{c}__qz"
        df[zname] = (df[c].values - m) / s
        new_cols.append(zname)

    return new_cols

def impute_clip(df: pd.DataFrame, feats: List[str]) -> None:
    """Median impute and robust clip per feature to reduce wild outliers."""
    for c in feats:
        col = df[c]
        med = np.nanmedian(col)
        df[c] = np.where(np.isnan(col), med, col)
        # clip to robust range
        q1, q99 = np.percentile(df[c], [1,99])
        if not math.isfinite(q1) or not math.isfinite(q99) or q1==q99:
            continue
        df[c] = np.clip(df[c], q1, q99)

def accuracy_oof(oof: np.ndarray, y: np.ndarray, thr: float) -> float:
    return accuracy_score(y, (oof >= thr).astype(int))

# -----------------------------
# Hyperparameter sampling
# -----------------------------

def sample_lgb_params(seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    return dict(
        n_estimators=rng.integers(300, 900),
        learning_rate=10**rng.uniform(-2.0, -1.1),  # ~0.01..0.079
        num_leaves=int(rng.integers(31, 127)),
        max_depth=int(rng.integers(4, 10)),
        feature_fraction=float(rng.uniform(0.7, 0.95)),
        bagging_fraction=float(rng.uniform(0.7, 0.95)),
        bagging_freq=int(rng.integers(1, 7)),
        reg_alpha=float(10**rng.uniform(-3, -0.3)),
        reg_lambda=float(10**rng.uniform(-3, -0.1)),
        min_child_samples=int(rng.integers(10, 60)),
        objective="binary",
        random_state=seed,
        n_jobs=-1
    )

def sample_xgb_params(seed: int, pos_weight: float) -> Dict:
    rng = np.random.default_rng(seed)
    return dict(
        n_estimators=int(rng.integers(300, 900)),
        learning_rate=float(10**rng.uniform(-2.0, -1.1)),  # ~0.01..0.079
        max_depth=int(rng.integers(4, 9)),
        subsample=float(rng.uniform(0.7, 0.95)),
        colsample_bytree=float(rng.uniform(0.7, 0.95)),
        reg_alpha=float(10**rng.uniform(-3, -0.3)),
        reg_lambda=float(10**rng.uniform(-3, -0.1)),
        min_child_weight=float(10**rng.uniform(-1.0, 1.0)),  # ~0.1..10
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=pos_weight
    )

# -----------------------------
# Core training per seed
# -----------------------------

@dataclass
class FoldResult:
    oof_lgb: np.ndarray
    oof_xgb: np.ndarray
    test_lgb: np.ndarray
    test_xgb: np.ndarray

def train_one_seed(X, y, g, X_test, seed: int, folds: int, lgb_params, xgb_params) -> FoldResult:
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, g), 1):
        log(f"[seed {seed}] Fold {fold}/{folds}")

        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb.predict_proba(X_va)[:,1]
        test_lgb += lgb.predict_proba(X_test)[:,1] / folds

        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_va)[:,1]
        test_xgb += xgb.predict_proba(X_test)[:,1] / folds

        log(f"  [Fold {fold}] mean p(LGB)={oof_lgb[val_idx].mean():.4f} | p(XGB)={oof_xgb[val_idx].mean():.4f}")

    return FoldResult(oof_lgb, oof_xgb, test_lgb, test_xgb)

# -----------------------------
# Weight + threshold search
# -----------------------------

def search_blend_and_threshold(oof_stacks: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    oof_stacks: shape [n_samples, n_models]
    returns: (best_weights, best_threshold, best_acc)
    """
    grids = np.linspace(0,1,11)
    best = (-1.0, None, None)
    ths = np.linspace(0.40, 0.60, 81)

    # weights for 2 or 3 columns
    m = oof_stacks.shape[1]
    if m == 2:
        for w0 in grids:
            w = np.array([w0, 1-w0])
            p = oof_stacks @ w
            for t in ths:
                acc = accuracy_score(y, (p >= t).astype(int))
                if acc > best[0]:
                    best = (acc, w, t)
    else:
        for w0 in grids:
            for w1 in grids:
                if w0 + w1 <= 1:
                    w = np.array([w0, w1, 1 - w0 - w1])
                    p = oof_stacks @ w
                    for t in ths:
                        acc = accuracy_score(y, (p >= t).astype(int))
                        if acc > best[0]:
                            best = (acc, w, t)

    return best[1], float(best[2]), float(best[0])

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max_inter_pairs", type=int, default=60)
    args = ap.parse_args()

    log(f"[INFO] Reading {args.train} / {args.test}")
    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)

    assert "id" in sample.columns and "relevance" in sample.columns, "Sample must have columns: id,relevance"
    assert "id" in test.columns, "Test must have 'id'"
    assert "relevance" in train.columns, "Train must have 'relevance'"
    assert "query_id" in train.columns, "Train must have 'query_id' for GroupKFold"

    target = "relevance"
    group_col = "query_id"
    drop_cols = ["id", "url_id", group_col, target]

    # Ensure test order matches sample IDs (to be safe)
    # We'll merge sample IDs -> test rows to guarantee final CSV aligns to sample
    test = sample[["id"]].merge(test, on="id", how="left")

    # Base numeric features
    base_feats = safe_numeric_cols(train, drop_cols=drop_cols)
    log(f"[INFO] Base numeric features: {len(base_feats)}")

    # Build a combined frame just to create engineering consistently, then split back
    train["_is_train"] = 1
    test["_is_train"] = 0
    df_all = pd.concat([train[["_is_train", group_col] + base_feats], test[["_is_train", group_col] + base_feats]], axis=0, ignore_index=True)

    # Impute/clip base features first
    impute_clip(df_all, base_feats)

    # Add interactions (using whole df_all X only — no target)
    inter_cols = add_interactions(df_all, base_feats, max_pairs=args.max_inter_pairs)
    log(f"[INFO] Added interactions: {len(inter_cols)}")

    # Add per-query features (rank/z-score) — uses only X
    pq_cols = add_per_query_features(df_all, group_col, base_feats)
    log(f"[INFO] Added per-query features: {len(pq_cols)}")

    # Final feature set
    feat_cols = base_feats + inter_cols + pq_cols
    log(f"[INFO] Total features: {len(feat_cols)}")

    # Split back
    df_train = df_all.loc[df_all["_is_train"]==1].reset_index(drop=True)
    df_test  = df_all.loc[df_all["_is_train"]==0].reset_index(drop=True)

    X = df_train[feat_cols].copy()
    y = train[target].astype(int).values
    g = train[group_col].values
    X_test = df_test[feat_cols].copy()

    # Hyperparameter search across trials & seeds
    best_global = {
        "acc": -1.0,
        "weights": None,
        "thr": 0.5,
        "seed": None,
        "lgb_params": None,
        "xgb_params": None,
        "oof": None,
        "test": None
    }

    pos_weight = (y==0).sum() / max((y==1).sum(), 1)

    for trial in range(1, args.trials+1):
        for seed in args.seeds:
            set_seed(seed)
            lgb_params = sample_lgb_params(seed + trial*1000)
            xgb_params = sample_xgb_params(seed + trial*2000, pos_weight)

            log(f"\n[TRIAL {trial}/{args.trials}] seed={seed}")
            log(f"  LGB params: {lgb_params}")
            log(f"  XGB params: {xgb_params}")

            fold_res = train_one_seed(X, y, g, X_test, seed, args.folds, lgb_params, xgb_params)

            # Meta learner on OOF of both models
            stack_train = np.vstack([fold_res.oof_lgb, fold_res.oof_xgb]).T
            meta = LogisticRegression(max_iter=3000)
            meta.fit(stack_train, y)
            oof_meta = meta.predict_proba(stack_train)[:,1]

            # Stack test
            stack_test = np.vstack([fold_res.test_lgb, fold_res.test_xgb]).T
            test_meta = meta.predict_proba(stack_test)[:,1]

            # Blend search across (lgb, xgb, meta)
            oof_stack3 = np.column_stack([fold_res.oof_lgb, fold_res.oof_xgb, oof_meta])
            best_w, best_thr, best_acc = search_blend_and_threshold(oof_stack3, y)

            log(f"  -> best OOF acc={best_acc:.5f} @ thr={best_thr:.4f} | weights={best_w}")

            if best_acc > best_global["acc"]:
                test_blend = (np.column_stack([fold_res.test_lgb, fold_res.test_xgb, test_meta]) @ best_w)
                best_global.update({
                    "acc": best_acc,
                    "weights": best_w.tolist(),
                    "thr": best_thr,
                    "seed": seed,
                    "lgb_params": lgb_params,
                    "xgb_params": xgb_params,
                    "oof": oof_stack3.tolist(),   # for trace (optional)
                    "test": test_blend.tolist()
                })

    log("\n[FINAL] Best OOF accuracy: {:.5f} (seed={})".format(best_global["acc"], best_global["seed"]))
    log("[FINAL] Weights: {}".format(best_global["weights"]))
    log("[FINAL] Threshold: {:.4f}".format(best_global["thr"]))

    # Build final predictions using the best trial
    test_blend = np.array(best_global["test"])
    final_pred = (test_blend >= best_global["thr"]).astype(int)

    # Write submission in sample order (id,relevance)
    sub = sample.copy()
    sub["relevance"] = final_pred
    sub.to_csv(args.out, index=False)
    log(f"[INFO] Wrote submission to {args.out} | rows={len(sub)}")

    # Save a compact summary
    summary = {
        "best_oof_acc": best_global["acc"],
        "best_threshold": best_global["thr"],
        "weights": best_global["weights"],
        "seed": best_global["seed"],
        "lgb_params": best_global["lgb_params"],
        "xgb_params": best_global["xgb_params"]
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[INFO] Saved summary -> {os.path.splitext(args.out)[0]}_summary.json")

if __name__ == "__main__":
    main()
