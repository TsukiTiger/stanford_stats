#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flip_with_agreement.py

Flip rows of an existing submission only where **two models agree** the label is wrong.
- Feature engineering (impute/clip, per-query z-scores & ranks, light interactions)
- Models: XGBoost + CatBoost
- Agreement-only flips, asymmetric margins, per-query cap, budget sweep
- Writes candidates and a flip report (indexes in base order + ids)

Meter: Public leaderboard **accuracy** (you submit the CSVs and pick the best).
"""

import argparse, os, json, math
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

def log(m): print(m, flush=True)

def numeric_features(df: pd.DataFrame, drop: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in drop]

def impute_clip(df: pd.DataFrame, feats: List[str]):
    for c in feats:
        v = df[c].to_numpy(copy=False)
        med = np.nanmedian(v)
        v = np.where(np.isnan(v), med, v)
        q1, q99 = np.percentile(v, [1, 99])
        if np.isfinite(q1) and np.isfinite(q99) and q1 < q99:
            v = np.clip(v, q1, q99)
        df[c] = v

def add_interactions(df: pd.DataFrame, base_cols: List[str], max_pairs=40) -> List[str]:
    if len(base_cols) < 2: return []
    var_rank = df[base_cols].var().sort_values(ascending=False).index.tolist()
    top = var_rank[:min(len(var_rank), max(8, int(len(var_rank)*0.5)))]
    pairs = []
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            pairs.append((top[i], top[j]))
    rng = np.random.default_rng(13); rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    new_cols = []
    for a,b in pairs:
        df[f"{a}__plus__{b}"]  = df[a] + df[b]
        df[f"{a}__minus__{b}"] = df[a] - df[b]
        df[f"{a}__times__{b}"] = df[a] * df[b]
        eps = 1e-6 * (df[b].abs().mean() + 1)
        df[f"{a}__ratio__{b}"] = df[a] / (df[b].abs() + eps)
        new_cols += [f"{a}__plus__{b}", f"{a}__minus__{b}", f"{a}__times__{b}", f"{a}__ratio__{b}"]
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

def select_with_caps(order_idx: np.ndarray, margins: np.ndarray, qids: np.ndarray,
                     max_per_query: int, budget: int) -> np.ndarray:
    """Greedy pick by descending margin with per-query cap."""
    order = order_idx[np.argsort(-margins[order_idx])]
    used = defaultdict(int)
    chosen = []
    for i in order:
        q = qids[i]
        if used[q] < max_per_query:
            chosen.append(i)
            used[q] += 1
            if len(chosen) >= budget:
                break
    return np.array(chosen, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out_prefix", default="flip_agree")
    ap.add_argument("--budgets", type=int, nargs="+", default=[60,80,100,120,140])
    ap.add_argument("--min_margin_up", type=float, default=0.055)   # 0->1
    ap.add_argument("--min_margin_down", type=float, default=0.075) # 1->0 a bit stricter
    ap.add_argument("--max_per_query", type=int, default=1)
    ap.add_argument("--use_gpu", type=int, default=0, help="1 = GPU for both models if available")
    args = ap.parse_args()

    train  = pd.read_csv(args.train)
    test   = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    base   = pd.read_csv(args.base)

    assert {"id","relevance"}.issubset(base.columns)
    assert {"id","relevance"}.issubset(sample.columns)
    assert {"id","query_id"}.issubset(test.columns)
    assert {"id","relevance","query_id"}.issubset(train.columns)

    # Align test to base order
    test_ord = base[["id"]].merge(test, on="id", how="left")
    assert len(test_ord) == len(base), "ID mismatch between base and test"
    base_y = base["relevance"].astype(int).to_numpy()
    qids   = test_ord["query_id"].to_numpy()

    # Features
    drop = ["id","url_id","query_id","relevance"]
    base_feats = numeric_features(train, drop)
    if not base_feats:
        raise RuntimeError("No numeric features found in training set.")

    train["_is_train"] = 1
    test_ord["_is_train"] = 0
    Xall = pd.concat([train[["_is_train","query_id"] + base_feats],
                      test_ord[["_is_train","query_id"] + base_feats]], axis=0, ignore_index=True)

    impute_clip(Xall, base_feats)
    inter_cols = add_interactions(Xall, base_feats, max_pairs=40)
    pq_cols    = add_query_features(Xall, "query_id", base_feats)
    feats = base_feats + inter_cols + pq_cols

    X_tr = Xall[Xall["_is_train"]==1].reset_index(drop=True)
    X_te = Xall[Xall["_is_train"]==0].reset_index(drop=True)
    y    = train["relevance"].astype(int).to_numpy()

    # --------- Train models ----------
    # XGB
    xgb_params = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.2, reg_lambda=0.3, min_child_weight=1.0,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        scale_pos_weight=float((y==0).sum() / max((y==1).sum(),1)),
        tree_method="gpu_hist" if args.use_gpu else "hist",
        predictor="gpu_predictor" if args.use_gpu else "auto"
    )
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr[feats].values, y)
    p_x = xgb.predict_proba(X_te[feats].values)[:,1]

    # CatBoost
    cat = CatBoostClassifier(
        iterations=1200, learning_rate=0.035, depth=6,
        l2_leaf_reg=3.0, loss_function="Logloss",
        random_seed=42, verbose=False,
        task_type="GPU" if args.use_gpu else "CPU"
    )
    cat.fit(Pool(X_tr[feats], y))
    p_c = cat.predict_proba(Pool(X_te[feats]))[:,1]

    # Agreement only: model class is the same AND different from base
    x_pred = (p_x >= 0.5).astype(int)
    c_pred = (p_c >= 0.5).astype(int)
    agree_pred = (x_pred == c_pred).astype(int)
    agree_class = x_pred  # == c_pred

    # margins for flipping if models agree against base
    # if base==0 and agree_class==1 -> candidate 0->1 with margin = avg(prob)
    # if base==1 and agree_class==0 -> candidate 1->0 with margin = avg(1-prob)
    p_avg = 0.5*(p_x + p_c)

    cand_up = np.where((base_y==0) & (agree_pred==1) & (agree_class==1))[0]
    cand_dn = np.where((base_y==1) & (agree_pred==1) & (agree_class==0))[0]

    margin_up = p_avg  # for 0->1
    margin_dn = 1.0 - p_avg  # for 1->0

    # apply min margins
    cand_up = cand_up[margin_up[cand_up] >= args.min_margin_up]
    cand_dn = cand_dn[margin_dn[cand_dn] >= args.min_margin_down]

    log(f"[INFO] Agreement candidates 0->1: {len(cand_up)}  (min_up={args.min_margin_up})")
    log(f"[INFO] Agreement candidates 1->0: {len(cand_dn)}  (min_dn={args.min_margin_down})")

    reports = []
    for B in args.budgets:
        total = len(cand_up) + len(cand_dn)
        if total == 0:
            chosen = np.array([], dtype=int)
        else:
            frac_up = len(cand_up) / total
            bud_up  = max(0, int(round(B * frac_up)))
            bud_dn  = max(0, B - bud_up)

            pick_up = select_with_caps(cand_up, margin_up, qids, args.max_per_query, bud_up)

            used_q = set(qids[i] for i in pick_up)
            mask_dn = np.array([qids[i] not in used_q for i in cand_dn])
            cand_dn_f = cand_dn[mask_dn]

            pick_dn = select_with_caps(cand_dn_f, margin_dn, qids, args.max_per_query, bud_dn)

            chosen = np.concatenate([pick_up, pick_dn], axis=0)

        flipped = base_y.copy()
        flipped[chosen] = 1 - flipped[chosen]

        # Write submission in sample order
        sub = sample.copy()
        tmp = pd.DataFrame({"id": base["id"], "relevance": flipped})
        sub = sub.merge(tmp, on="id", how="left", suffixes=("_tmpl",""))
        sub["relevance"] = sub["relevance"].fillna(sub["relevance_tmpl"]).astype(int)
        sub = sub[["id","relevance"]]
        out_path = f"{args.out_prefix}_B{B}_cap{args.max_per_query}_up{args.min_margin_up:.3f}_dn{args.min_margin_down:.3f}.csv"
        sub.to_csv(out_path, index=False)
        log(f"[WRITE] {out_path} | flips={len(chosen)}")

        reports.append({
            "file": os.path.basename(out_path),
            "budget": int(B),
            "max_per_query": int(args.max_per_query),
            "min_margin_up": float(args.min_margin_up),
            "min_margin_down": float(args.min_margin_down),
            "num_flipped": int(len(chosen)),
            "first_80_indexes": [int(i) for i in chosen[:80]],
            "first_80_ids": base["id"].iloc[chosen[:80]].tolist()
        })

    with open(f"{args.out_prefix}_report.json", "w") as f:
        json.dump({
            "note": "Indexes are 0-based in BASE submission order.",
            "base_file": os.path.basename(args.base),
            "budgets": args.budgets,
            "max_per_query": args.max_per_query,
            "min_margin_up": args.min_margin_up,
            "min_margin_down": args.min_margin_down,
            "reports": reports
        }, f, indent=2)
    log(f"[INFO] Saved report -> {args.out_prefix}_report.json")

if __name__ == "__main__":
    main()
