#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked GPU model (LGBM + XGB + CatBoost) with query-aware features, OOF weight+threshold search,
and optional Top-K per-query blending. Optimizes OOF **accuracy** (same leaderboard metric).

Usage:
python stack_strong_gpu.py \
  --train ./data/training.csv \
  --test ./data/test.csv \
  --sample ./data/sample_submission.csv \
  --out submission.csv \
  --folds 5 \
  --seeds 42 777 \
  --trials 24 \
  --max_inter_pairs 60 \
  --topk 1 \
  --alpha_grid 0.0 0.2 0.3 0.4
"""

import argparse, json, math, os, random
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# GPU models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

# --------------------- utils ---------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def log(msg: str): print(msg, flush=True)

def numeric_features(df: pd.DataFrame, drop: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in drop]

def impute_and_clip(df: pd.DataFrame, feats: List[str]):
    for c in feats:
        col = df[c]
        med = np.nanmedian(col)
        v = np.where(np.isnan(col), med, col)
        q1, q99 = np.percentile(v, [1, 99])
        if math.isfinite(q1) and math.isfinite(q99) and q1 < q99:
            v = np.clip(v, q1, q99)
        df[c] = v

def add_interactions(df: pd.DataFrame, base_cols: List[str], max_pairs=60) -> List[str]:
    if len(base_cols) < 2: return []
    var_rank = df[base_cols].var().sort_values(ascending=False).index.tolist()
    top = var_rank[:min(len(var_rank), max(8, int(len(var_rank)*0.5)))]
    pairs = []
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            pairs.append((top[i], top[j]))
    rng = np.random.default_rng(13)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    new_cols = []
    for a,b in pairs:
        c1 = f"{a}__plus__{b}"; df[c1] = df[a] + df[b]
        c2 = f"{a}__minus__{b}"; df[c2] = df[a] - df[b]
        c3 = f"{a}__times__{b}"; df[c3] = df[a] * df[b]
        eps = 1e-6 * (df[b].abs().mean() + 1)
        c4 = f"{a}__ratio__{b}"; df[c4] = df[a] / (df[b].abs() + eps)
        new_cols += [c1,c2,c3,c4]
    return new_cols

def add_query_features(df: pd.DataFrame, qcol: str, feats: List[str]) -> List[str]:
    g = df.groupby(qcol, observed=True)
    new_cols = []
    # ranks
    for c in feats:
        r = f"{c}__qrank"
        df[r] = g[c].rank(method="first", ascending=True).astype(np.float32)
        new_cols.append(r)
    # z-scores
    stats = g[feats].agg(['mean','std'])
    for c in feats:
        m = stats[(c,'mean')].reindex(df[qcol]).to_numpy()
        s = stats[(c,'std')].reindex(df[qcol]).to_numpy()
        s = np.where((s==0) | (~np.isfinite(s)), 1.0, s)
        z = f"{c}__qz"
        df[z] = (df[c].to_numpy() - m) / s
        new_cols.append(z)
    # is max/min
    for c in feats:
        mx = f"{c}__is_qmax"; mn = f"{c}__is_qmin"
        df[mx] = (df[c] == g[c].transform("max")).astype(np.int8)
        df[mn] = (df[c] == g[c].transform("min")).astype(np.int8)
        new_cols += [mx, mn]
    # group size
    df["__qsize"] = g[feats[0]].transform("size").astype(np.int32)
    new_cols.append("__qsize")
    return new_cols

def acc_at(p: np.ndarray, y: np.ndarray, thr: float) -> float:
    return accuracy_score(y, (p >= thr).astype(int))

def search_weights_and_threshold(oof_stack: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Grid search blend weights + threshold (optimizes OOF accuracy)."""
    grids = np.linspace(0, 1, 11)
    best = (-1.0, None, None)
    ths = np.linspace(0.40, 0.60, 81)
    m = oof_stack.shape[1]
    if m == 2:
        for w0 in grids:
            w = np.array([w0, 1-w0])
            p = oof_stack @ w
            for t in ths:
                a = acc_at(p, y, t)
                if a > best[0]: best = (a, w, t)
    else:
        for w0 in grids:
            for w1 in grids:
                if w0 + w1 <= 1:
                    w = np.array([w0, w1, 1-w0-w1])
                    p = oof_stack @ w
                    for t in ths:
                        a = acc_at(p, y, t)
                        if a > best[0]: best = (a, w, t)
    return best[1], float(best[2]), float(best[0])

def topk_override(scores: np.ndarray, qids: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(scores, dtype=int)
    df = pd.DataFrame({"qid": qids, "score": scores, "_i": np.arange(len(scores))})
    for q, grp in df.groupby("qid", sort=False):
        kk = min(k, len(grp))
        top_idx = grp["score"].nlargest(kk).index
        out[grp.loc[top_idx, "_i"].to_numpy()] = 1
    return out

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--trials", type=int, default=24, help="random trials per seed for param sampling (base models)")
    ap.add_argument("--max_inter_pairs", type=int, default=60)
    ap.add_argument("--topk", type=int, default=1, help="Top-K per query for blending (0 disables)")
    ap.add_argument("--alpha_grid", type=float, nargs="+", default=[0.0,0.2,0.3,0.4], help="blend weight with top-k mask")
    args = ap.parse_args()

    log(f"[INFO] Reading {args.train} / {args.test}")
    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)

    assert {"id","relevance","query_id"}.issubset(train.columns)
    assert {"id","query_id"}.issubset(test.columns)
    assert {"id","relevance"}.issubset(sample.columns)
    target, qcol = "relevance", "query_id"
    drop = ["id","url_id", qcol, target]

    # Align test to sample order (never trust raw order)
    test = sample[["id"]].merge(test, on="id", how="left")
    assert len(test) == len(sample)

    # Base feats
    base_feats = numeric_features(train, drop)
    log(f"[INFO] Base numeric features: {len(base_feats)}")

    # Build combined X-only frame for consistent featurization
    train["_is_train"] = 1
    test["_is_train"]  = 0
    Xall = pd.concat([train[["_is_train", qcol] + base_feats],
                      test [["_is_train", qcol] + base_feats]], axis=0, ignore_index=True)

    # Clean + engineer
    impute_and_clip(Xall, base_feats)
    inter_cols = add_interactions(Xall, base_feats, max_pairs=args.max_inter_pairs)
    pq_cols    = add_query_features(Xall, qcol, base_feats)
    feats = base_feats + inter_cols + pq_cols
    log(f"[INFO] Features: base={len(base_feats)} inter={len(inter_cols)} query={len(pq_cols)} TOTAL={len(feats)}")

    # Split back
    X_tr = Xall[Xall["_is_train"]==1].reset_index(drop=True)
    X_te = Xall[Xall["_is_train"]==0].reset_index(drop=True)
    y = train[target].astype(int).to_numpy()
    g = train[qcol].to_numpy()
    qid_te = test[qcol].to_numpy()

    # Training holders (we'll keep best per seed and average test)
    oof_lgb_all, oof_xgb_all, oof_cat_all = [], [], []
    te_lgb_all,  te_xgb_all,  te_cat_all  = [], [], []

    # pos weight for XGB
    pos_w = (y==0).sum() / max((y==1).sum(), 1)

    def sample_lgb(seed):
        rng = np.random.default_rng(seed)
        return dict(
            n_estimators=int(rng.integers(500, 1100)),
            learning_rate=float(10**rng.uniform(-2.0,-1.05)), # ~0.01..0.089
            num_leaves=int(rng.integers(31, 127)),
            max_depth=-1,
            min_child_samples=int(rng.integers(10, 60)),
            feature_fraction=float(rng.uniform(0.75, 0.95)),
            bagging_fraction=float(rng.uniform(0.7, 0.95)),
            bagging_freq=int(rng.integers(1, 7)),
            reg_alpha=float(10**rng.uniform(-3, -0.2)),
            reg_lambda=float(10**rng.uniform(-3, -0.1)),
            objective="binary",
            random_state=seed,
            # device="gpu"
        )

    def sample_xgb(seed):
        rng = np.random.default_rng(seed)
        return dict(
            n_estimators=int(rng.integers(500, 1100)),
            learning_rate=float(10**rng.uniform(-2.0,-1.05)),
            max_depth=int(rng.integers(5, 9)),
            subsample=float(rng.uniform(0.75, 0.95)),
            colsample_bytree=float(rng.uniform(0.75, 0.95)),
            reg_alpha=float(10**rng.uniform(-3, -0.2)),
            reg_lambda=float(10**rng.uniform(-3, -0.1)),
            min_child_weight=float(10**rng.uniform(-1.0, 1.0)),
            eval_metric="logloss",
            random_state=seed,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            n_jobs=-1,
            scale_pos_weight=float(pos_w)
        )

    # ========== Train across seeds ==========
    for seed in args.seeds:
        set_seed(seed)
        log(f"\n[SEED {seed}]")
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)

        # We'll keep the single best trial (by OOF acc after stacking) for this seed
        best_for_seed = {"acc": -1, "oof": None, "te": None}

        for trial in range(1, args.trials+1):
            log(f"[seed {seed}] trial {trial}/{args.trials}")
            p_lgb = sample_lgb(seed + 1000*trial)
            p_xgb = sample_xgb(seed + 2000*trial)

            oof_lgb = np.zeros(len(X_tr)); te_lgb = np.zeros(len(X_te))
            oof_xgb = np.zeros(len(X_tr)); te_xgb = np.zeros(len(X_te))
            oof_cat = np.zeros(len(X_tr)); te_cat = np.zeros(len(X_te))

            for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_tr[feats], y, g), 1):
                Xtr, Xva = X_tr.iloc[trn_idx][feats], X_tr.iloc[val_idx][feats]
                ytr, yva = y[trn_idx], y[val_idx]

                # LGB
                lgb = LGBMClassifier(**p_lgb)
                lgb.fit(Xtr, ytr)
                oof_lgb[val_idx] = lgb.predict_proba(Xva)[:,1]
                te_lgb += lgb.predict_proba(X_te[feats])[:,1] / args.folds

                # XGB
                xgb = XGBClassifier(**p_xgb)
                xgb.fit(Xtr, ytr)
                oof_xgb[val_idx] = xgb.predict_proba(Xva)[:,1]
                te_xgb += xgb.predict_proba(X_te[feats])[:,1] / args.folds

                # CatBoost (fixed strong params for stability)
                cat = CatBoostClassifier(
                    iterations=1200, learning_rate=0.035, depth=6,
                    l2_leaf_reg=3.0, loss_function="Logloss",
                    random_seed=seed, verbose=False, task_type="GPU"
                )
                cat.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva))
                oof_cat[val_idx] = cat.predict_proba(Xva)[:,1]
                te_cat += cat.predict_proba(X_te[feats])[:,1] / args.folds

            # Meta on OOF (base probs + simple query meta-features)
            qsize = X_tr["__qsize"].to_numpy()
            # Include rank of each base model prob within query for meta
            df_meta = pd.DataFrame({
                "qid": g,
                "lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat,
                "qsize": qsize
            })
            for c in ["lgb","xgb","cat"]:
                df_meta[c+"_r"] = df_meta.groupby("qid", observed=True)[c].rank(method="first", ascending=False).astype(np.float32)
            meta_feats = ["lgb","xgb","cat","qsize","lgb_r","xgb_r","cat_r"]

            scaler_lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=4000))])
            scaler_lr.fit(df_meta[meta_feats].values, y)
            oof_meta = scaler_lr.predict_proba(df_meta[meta_feats].values)[:,1]

            # Apply same meta on test
            df_meta_te = pd.DataFrame({
                "qid": qid_te,
                "lgb": te_lgb, "xgb": te_xgb, "cat": te_cat
            })
            # approximate ranks within test query
            for c in ["lgb","xgb","cat"]:
                df_meta_te[c+"_r"] = df_meta_te.groupby("qid", observed=True)[c].rank(method="first", ascending=False).astype(np.float32)
            df_meta_te["qsize"] = pd.Series(qid_te).map(pd.Series(qid_te).value_counts()).values
            te_meta = scaler_lr.predict_proba(df_meta_te[meta_feats].values)[:,1]

            # Blend search (base & meta)
            base_oof = np.column_stack([oof_lgb, oof_xgb, oof_cat])
            base_te  = np.column_stack([te_lgb,  te_xgb,  te_cat])
            w_base, thr_base, acc_base = search_weights_and_threshold(base_oof, y)
            p_base_oof = base_oof @ w_base
            p_base_te  = base_te  @ w_base

            oof_stack2 = np.column_stack([p_base_oof, oof_meta])
            te_stack2  = np.column_stack([p_base_te,  te_meta])
            w2, thr2, acc2 = search_weights_and_threshold(oof_stack2, y)
            # acc2 is OOF accuracy after stacking (this is our selection criterion)
            if acc2 > best_for_seed["acc"]:
                best_for_seed.update({
                    "acc": acc2,
                    "oof": (oof_lgb, oof_xgb, oof_cat, oof_meta, p_base_oof),
                    "te":  (te_lgb,  te_xgb,  te_cat,  te_meta,  p_base_te),
                    "w_base": w_base.tolist(),
                    "thr_base": float(thr_base),
                    "w2": w2.tolist(),
                    "thr2": float(thr2)
                })
                log(f"  [seed {seed}] trial {trial} NEW BEST OOF acc={acc2:.5f} | thr={thr2:.4f} | w2={w2} | w_base={w_base}")

        # keep best-of-seed predictions
        te_lgb_best, te_xgb_best, te_cat_best, te_meta_best, p_base_te_best = best_for_seed["te"]
        oof_lgb_best, oof_xgb_best, oof_cat_best, oof_meta_best, p_base_oof_best = best_for_seed["oof"]

        oof_lgb_all.append(oof_lgb_best); oof_xgb_all.append(oof_xgb_best); oof_cat_all.append(oof_cat_best)
        te_lgb_all.append(te_lgb_best);   te_xgb_all.append(te_xgb_best);   te_cat_all.append(te_cat_best)

    # Average across seeds
    oof_lgb = np.mean(oof_lgb_all, axis=0); te_lgb = np.mean(te_lgb_all, axis=0)
    oof_xgb = np.mean(oof_xgb_all, axis=0); te_xgb = np.mean(te_xgb_all, axis=0)
    oof_cat = np.mean(oof_cat_all, axis=0); te_cat = np.mean(te_cat_all, axis=0)

    # Rebuild meta after seed-averaging
    df_meta = pd.DataFrame({"qid": g, "lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat})
    df_meta["qsize"] = X_tr["__qsize"].to_numpy()
    for c in ["lgb","xgb","cat"]:
        df_meta[c+"_r"] = df_meta.groupby("qid", observed=True)[c].rank(method="first", ascending=False).astype(np.float32)
    meta_feats = ["lgb","xgb","cat","qsize","lgb_r","xgb_r","cat_r"]

    scaler_lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=4000))])
    scaler_lr.fit(df_meta[meta_feats].values, y)
    oof_meta = scaler_lr.predict_proba(df_meta[meta_feats].values)[:,1]

    df_meta_te = pd.DataFrame({"qid": qid_te, "lgb": te_lgb, "xgb": te_xgb, "cat": te_cat})
    for c in ["lgb","xgb","cat"]:
        df_meta_te[c+"_r"] = df_meta_te.groupby("qid", observed=True)[c].rank(method="first", ascending=False).astype(np.float32)
    df_meta_te["qsize"] = pd.Series(qid_te).map(pd.Series(qid_te).value_counts()).values
    te_meta = scaler_lr.predict_proba(df_meta_te[meta_feats].values)[:,1]

    # Final blend search (base->meta)
    base_oof = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    base_te  = np.column_stack([te_lgb,  te_xgb,  te_cat])
    w_base, thr_base, acc_base = search_weights_and_threshold(base_oof, y)
    p_base_oof = base_oof @ w_base
    p_base_te  = base_te  @ w_base

    oof_stack2 = np.column_stack([p_base_oof, oof_meta])
    te_stack2  = np.column_stack([p_base_te,  te_meta])
    w2, thr2, acc2 = search_weights_and_threshold(oof_stack2, y)
    log(f"[FINAL STACK] OOF acc={acc2:.5f} @ thr={thr2:.4f} | w(base,meta)={w2} | w_base={w_base}")

    # Predict test probabilities
    test_scores = te_stack2 @ w2
    best_thr = thr2
    best_oof = acc2

    # Optional query-aware Top-K blending (test-time only)
    final_pred = (test_scores >= best_thr).astype(int)
    if args.topk > 0 and len(args.alpha_grid) > 0:
        best_pp = (0.0, None, None)  # (OOF acc, alpha, chosen strategy)
        # Build OOF analogue of topk (using OOF probs + true query ids)
        oof_scores = oof_stack2 @ w2
        oof_topk = topk_override(oof_scores, g, k=args.topk)
        # Try blend alphas and pick the one that maximizes OOF accuracy
        for alpha in args.alpha_grid:
            # blend: (1-alpha)*hard_ml + alpha*topk
            hard_ml = (oof_scores >= best_thr).astype(int)
            blended = ((1 - alpha)*hard_ml + alpha*oof_topk) >= 0.5
            acc_pp = accuracy_score(y, blended.astype(int))
            if acc_pp > best_pp[0]:
                best_pp = (acc_pp, alpha, "topk")
        log(f"[POSTPROC] OOF acc with best alpha={best_pp[1]:.2f} -> {best_pp[0]:.5f} (plain {best_oof:.5f})")
        # Apply same alpha to test
        if best_pp[1] is not None:
            topk_mask_te = topk_override(test_scores, qid_te, k=args.topk)
            hard_ml_te = (test_scores >= best_thr).astype(int)
            blended_te = ((1 - best_pp[1])*hard_ml_te + best_pp[1]*topk_mask_te) >= 0.5
            final_pred = blended_te.astype(int)

    # Write submission in sample order
    sub = sample.copy()
    sub["relevance"] = final_pred
    sub.to_csv(args.out, index=False)
    log(f"[WRITE] {args.out}  rows={len(sub)}")

    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({
            "best_oof_acc": float(best_oof),
            "best_threshold": float(best_thr),
            "weights_base": [float(x) for x in w_base],
            "weights_base_meta": [float(x) for x in w2],
            "features_total": int(len(feats)),
            "topk_used": int(args.topk),
            "alpha_grid": [float(x) for x in args.alpha_grid]
        }, f, indent=2)
    log("[DONE]")

if __name__ == "__main__":
    main()
