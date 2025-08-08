#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flip rows of an existing submission using a trained model's high-confidence disagreements.

- Aligns to base submission order by `id`
- Feature engineering: robust impute/clip, per-query z-scores & ranks, simple interactions
- XGBoost scorer (GPU optional); CPU works fine
- Produces multiple flipped submissions for Top-N = [100, 250, 500] (configurable)
- Also supports min-margin so only confident flips happen
- Emits a JSON with flipped row indexes (0-based in base order) + ids

Usage:
python flip_with_model.py \
  --train ./data/training.csv \
  --test ./data/test.csv \
  --sample ./data/sample_submission.csv \
  --base ./submission_05.csv \
  --out_prefix flipped_from05 \
  --topn_list 100 250 500 \
  --min_margin 0.04 \
  --use_gpu 0
"""

import argparse, os, json, math
import numpy as np
import pandas as pd
from typing import List

from xgboost import XGBClassifier

def log(m): print(m, flush=True)

def numeric_features(df: pd.DataFrame, drop: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in drop]

def impute_clip(df: pd.DataFrame, feats: List[str]):
    for c in feats:
        v = df[c].to_numpy()
        med = np.nanmedian(v)
        v = np.where(np.isnan(v), med, v)
        q1, q99 = np.percentile(v, [1, 99])
        if math.isfinite(q1) and math.isfinite(q99) and q1 < q99:
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
        c1 = f"{a}__plus__{b}";  df[c1] = df[a] + df[b]
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out_prefix", default="flipped")
    ap.add_argument("--topn_list", type=int, nargs="+", default=[100,250,500])
    ap.add_argument("--min_margin", type=float, default=0.0)
    ap.add_argument("--use_gpu", type=int, default=0)
    args = ap.parse_args()

    # Load
    train  = pd.read_csv(args.train)
    test   = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    base   = pd.read_csv(args.base)

    assert {"id","relevance"}.issubset(base.columns)
    assert {"id","relevance"}.issubset(sample.columns)
    assert {"id","query_id"}.issubset(test.columns)
    assert {"id","relevance","query_id"}.issubset(train.columns)

    # Align test to base order (so indexes we report are in base order)
    test_ord = base[["id"]].merge(test, on="id", how="left")
    assert len(test_ord) == len(base), "ID mismatch between base and test"

    # Base labels (0/1) in base order
    base_y = base["relevance"].astype(int).to_numpy()
    qids   = test_ord["query_id"].to_numpy()

    # Build features (on combined frame for consistent transforms)
    drop = ["id","url_id","query_id","relevance"]
    base_feats = numeric_features(train, drop)
    if not base_feats:
        raise RuntimeError("No numeric features found in training set.")

    # Put train/test together for X-only transforms
    train["_is_train"] = 1
    test_ord["_is_train"] = 0
    Xall = pd.concat([train[["_is_train","query_id"] + base_feats],
                      test_ord[["_is_train","query_id"] + base_feats]],
                     axis=0, ignore_index=True)

    impute_clip(Xall, base_feats)
    inter_cols = add_interactions(Xall, base_feats, max_pairs=40)
    pq_cols    = add_query_features(Xall, "query_id", base_feats)
    feats = base_feats + inter_cols + pq_cols

    X_tr = Xall[Xall["_is_train"]==1].reset_index(drop=True)
    X_te = Xall[Xall["_is_train"]==0].reset_index(drop=True)
    y    = train["relevance"].astype(int).to_numpy()

    # XGBoost model (fast, solid). GPU optional.
    params = dict(
        n_estimators=700, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.2, reg_lambda=0.3, min_child_weight=1.0,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
        scale_pos_weight=float((y==0).sum() / max((y==1).sum(),1))
    )
    if args.use_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist")

    clf = XGBClassifier(**params)
    clf.fit(X_tr[feats].values, y)
    test_prob = clf.predict_proba(X_te[feats].values)[:,1]

    # Compute disagreement margin vs base labels
    # If base=0 -> margin = prob (confidence to flip to 1)
    # If base=1 -> margin = (1 - prob) (confidence to flip to 0)
    margin = np.where(base_y==0, test_prob, 1.0 - test_prob)

    # Rank by margin desc
    order = np.argsort(-margin)
    # Keep only those that actually want to flip (i.e., would change the label)
    want_flip = np.where(
        ((base_y==0) & (test_prob > 0.5)) | ((base_y==1) & (test_prob < 0.5))
    )[0]
    # Apply min-margin filter if requested
    if args.min_margin > 0:
        want_flip = want_flip[margin[want_flip] >= args.min_margin]
    # Reorder by margin desc
    want_flip = want_flip[np.argsort(-margin[want_flip])]

    log(f"[INFO] Candidates wanting flip = {len(want_flip)} (min_margin={args.min_margin})")

    reports = []
    for N in args.topn_list:
        idx_to_flip = want_flip[:N]
        flipped = base_y.copy()
        flipped[idx_to_flip] = 1 - flipped[idx_to_flip]

        # Write submission in sample order (id,relevance)
        sub = sample.copy()
        tmp = pd.DataFrame({"id": base["id"], "relevance": flipped})
        sub = sub.merge(tmp, on="id", how="left", suffixes=("_tmpl",""))
        sub["relevance"] = sub["relevance"].fillna(sub["relevance_tmpl"]).astype(int)
        sub = sub[["id","relevance"]]

        out_path = f"{args.out_prefix}_top{N}_m{args.min_margin:.3f}.csv"
        sub.to_csv(out_path, index=False)
        log(f"[WRITE] {out_path} | flips={len(idx_to_flip)}")

        # Collect report with indexes (0-based in base order) and ids
        flipped_ids = base["id"].iloc[idx_to_flip].tolist()
        reports.append({
            "file": os.path.basename(out_path),
            "topN": int(N),
            "min_margin": float(args.min_margin),
            "num_flipped": int(len(idx_to_flip)),
            "indexes_0based_in_base_order": idx_to_flip[:500].tolist(),  # cap for readability
            "ids": flipped_ids[:500]
        })

    with open(f"{args.out_prefix}_report.json", "w") as f:
        json.dump({
            "note": "Indexes are 0-based row positions in your BASE submission file.",
            "base_file": os.path.basename(args.base),
            "n_candidates": len(want_flip),
            "topn_runs": reports
        }, f, indent=2)
    log(f"[INFO] Saved report -> {args.out_prefix}_report.json")

if __name__ == "__main__":
    main()
