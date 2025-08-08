#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process an existing submission (e.g., submission_05.csv) to push accuracy up.

What it does:
  1) Aligns base submission to sample order.
  2) Exact train→test override if (query_id, url_id) appears in train (strong signal).
  3) Builds query-aware heuristic scores from standardized numeric features.
  4) Creates candidates using per-query Top-K + blend alpha, plus an optional kNN-ish similarity bump.
  5) Writes candidates and a flip-index list (row indexes in base submission order).

Meter (what we optimize on the site): **Leaderboard Accuracy**
Locally, we can't compute LB accuracy (no test labels), so we output multiple strong candidates.
You submit and report back accuracy; we’ll refine k / alpha based on your feedback.

Usage example:
python improve_from_submission.py \
  --train ./data/training.csv \
  --test ./data/test.csv \
  --sample ./data/sample_submission.csv \
  --base ./submission_05.csv \
  --out_prefix tuned_from05 \
  --k_list 1 2 \
  --alpha_list 0.25 0.30 0.35 \
  --use_knn 1
"""

import argparse, os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from math import isfinite

def log(m): print(m, flush=True)

def numeric_features(df, drop_cols):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in drop_cols]

def impute_clip(df, cols):
    # median-impute, then clip to [1,99] percentile to tame outliers
    for c in cols:
        v = df[c].to_numpy(copy=False)
        med = np.nanmedian(v)
        v = np.where(np.isnan(v), med, v)
        q1, q99 = np.percentile(v, [1, 99])
        if isfinite(q1) and isfinite(q99) and q1 < q99:
            v = np.clip(v, q1, q99)
        df[c] = v

def standardize_within_group(df, group_col, cols):
    g = df.groupby(group_col, observed=True)
    means = g[cols].transform("mean")
    stds  = g[cols].transform("std").replace(0, 1.0).fillna(1.0)
    Z = (df[cols] - means) / stds
    return Z

def build_heuristic_scores(test_df, qcol, feat_cols, base_binary):
    """
    Query-standardized weighted sum with tiny bias toward current base positives.
    """
    Z = standardize_within_group(test_df, qcol, feat_cols).to_numpy()
    # emphasize common signal-y columns if present (adjust names if yours differ)
    weights = np.ones(Z.shape[1], dtype=float)
    for i, c in enumerate(feat_cols):
        lc = c.lower()
        if any(sig in lc for sig in ["sig2","sig6","sig1","sig5","sig7","sig3"]):
            weights[i] = 1.5
    score = (Z * weights).sum(axis=1) + 0.05 * base_binary  # gentle nudge to base label
    return score

def topk_mask(scores, qids, k):
    out = np.zeros_like(scores, dtype=int)
    df = pd.DataFrame({"qid": qids, "score": scores, "_i": np.arange(len(scores))})
    for q, grp in df.groupby("qid", sort=False):
        kk = min(k, len(grp))
        top_idx = grp["score"].nlargest(kk).index
        out[grp.loc[top_idx, "_i"].to_numpy()] = 1
    return out

def build_knn_score(train_df, test_df, qcol, feat_cols, ycol="relevance", n_neighbors=25):
    """
    Lightweight similarity score per row using cosine on query-standardized features,
    but only within the SAME query_id to avoid leakage across queries.
    Returns a vector in [0,1] ~ fraction of positive neighbors.
    """
    # standardize per query separately in train and test (on their own groups)
    Ztr = standardize_within_group(train_df, qcol, feat_cols).to_numpy()
    Zte = standardize_within_group(test_df,  qcol, feat_cols).to_numpy()
    ytr = train_df[ycol].astype(int).to_numpy()
    qtr = train_df[qcol].to_numpy()
    qte = test_df[qcol].to_numpy()

    # Build per-query indices for fast lookup
    idx_by_q = defaultdict(list)
    for i, q in enumerate(qtr):
        idx_by_q[q].append(i)

    # Cosine similarity (normalized vectors)
    def l2norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X / n

    Ztrn = l2norm(Ztr)
    Zten = l2norm(Zte)

    scores = np.zeros(len(Zte), dtype=float)
    for i in range(len(Zte)):
        q = qte[i]
        tr_idx = idx_by_q.get(q, [])
        if not tr_idx:
            scores[i] = 0.5  # neutral if no same-query history
            continue
        Xq = Ztrn[tr_idx]
        yq = ytr[tr_idx]
        v  = Zten[i:i+1]  # (1, d)
        # cosine sims -> pick top neighbors
        sims = (Xq @ v.T).ravel()
        k = min(n_neighbors, len(sims))
        top = np.argpartition(-sims, k-1)[:k]
        pos_frac = yq[top].mean() if k > 0 else 0.5
        scores[i] = float(pos_frac)
    return scores

def exact_override_from_train(train_df, test_df, base_binary, qcol="query_id"):
    """
    If a (query_id, url_id) exact pair is present in train, override test label with train label.
    This is safe if the dataset truly reuses pairs; otherwise it won’t trigger often.
    Returns new labels and a mask indicating which rows were overridden.
    """
    if "url_id" not in train_df.columns or "url_id" not in test_df.columns:
        return base_binary.copy(), np.zeros_like(base_binary, dtype=bool)

    key = ["query_id", "url_id"]
    train_pairs = train_df[key + ["relevance"]].drop_duplicates()
    test_pairs  = test_df[key].copy()
    merged = test_pairs.merge(train_pairs, on=key, how="left")  # relevance_y from train
    overrides = merged["relevance"].values
    new_labels = base_binary.copy()
    mask = ~pd.isna(overrides)
    new_labels[mask] = overrides[mask].astype(int)
    return new_labels, mask

def write_submission(sample, ids, preds, out_path):
    sub = sample.copy()
    tmp = pd.DataFrame({"id": ids, "relevance": preds})
    sub = sub.merge(tmp, on="id", how="left", suffixes=("_tmpl",""))
    sub["relevance"] = sub["relevance"].fillna(sub["relevance_tmpl"]).astype(int)
    sub = sub[["id","relevance"]]
    sub.to_csv(out_path, index=False)
    return sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--base",   required=True)
    ap.add_argument("--out_prefix", default="tuned")
    ap.add_argument("--k_list", type=int, nargs="+", default=[1,2])
    ap.add_argument("--alpha_list", type=float, nargs="+", default=[0.25,0.30,0.35])
    ap.add_argument("--use_knn", type=int, default=1, help="1=use knn similarity bump")
    ap.add_argument("--neighbors", type=int, default=25)
    args = ap.parse_args()

    # Read inputs
    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    samp  = pd.read_csv(args.sample)
    base  = pd.read_csv(args.base)

    assert {"id","relevance"}.issubset(base.columns)
    assert {"id","relevance"}.issubset(samp.columns)
    assert {"id","query_id"}.issubset(test.columns)
    assert {"id","relevance","query_id"}.issubset(train.columns)

    # Align test to base order by id
    test_ord = base[["id"]].merge(test, on="id", how="left")
    assert len(test_ord) == len(base), "ID mismatch between base submission and test"

    # Base labels (binary) in base order
    base_y = base["relevance"].astype(int).to_numpy()
    qids   = test_ord["query_id"].to_numpy()

    # OPTIONAL: exact overrides from train if (query_id, url_id) pair repeats
    base_after_pair, pair_mask = exact_override_from_train(train, test_ord, base_y, qcol="query_id")
    if pair_mask.any():
        log(f"[PAIR OVERRIDE] Applied to {pair_mask.sum()} rows (from exact (query_id,url_id) matches)")

    # Minimal, robust feature set
    drop_cols = {"id","url_id","query_id","relevance"}
    feat_cols = numeric_features(test_ord, drop_cols=drop_cols)
    if not feat_cols:
        raise RuntimeError("No numeric features found in test set.")
    # Clean features
    impute_clip(test_ord, feat_cols)

    # Heuristic query-aware score
    heur_scores = build_heuristic_scores(test_ord, "query_id", feat_cols, base_binary=base_after_pair)

    # Optional kNN-ish similarity bump using train data (same query only)
    if args.use_knn:
        knn_scores = build_knn_score(train[["query_id","relevance"] + feat_cols],
                                     test_ord[["query_id"] + feat_cols],
                                     qcol="query_id", feat_cols=feat_cols,
                                     ycol="relevance", n_neighbors=args.neighbors)
        # combine scores (simple average)
        scores = 0.5 * heur_scores + 0.5 * knn_scores
    else:
        scores = heur_scores

    # Create candidates via Top-K + blend alpha wrt current base_after_pair
    out_files = []
    flip_reports = []
    for k in args.k_list:
        topk = topk_mask(scores, qids, k=k)  # 0/1 within each query
        for alpha in args.alpha_list:
            # blend with base predictions after exact-pair overrides:
            # final = round( (1-alpha)*base + alpha*topk )
            blended = ((1 - alpha) * base_after_pair + alpha * topk)
            final = (blended >= 0.5).astype(int)

            # write submission in sample order
            out_path = f"{args.out_prefix}_k{k}_a{alpha:.2f}.csv"
            sub = write_submission(samp, base["id"], final, out_path)
            out_files.append(out_path)

            # diff vs base (row indexes in base order)
            flips = np.where(final != base_y)[0].tolist()
            flip_reports.append({
                "file": out_path,
                "k": k,
                "alpha": alpha,
                "num_flips": len(flips),
                "first_50_indexes": flips[:50]
            })
            log(f"[WRITE] {out_path} | flips={len(flips)}")

    # Save a summary with flip lists
    meta = {
        "base_file": os.path.basename(args.base),
        "k_list": args.k_list,
        "alpha_list": args.alpha_list,
        "use_knn": int(args.use_knn),
        "neighbors": int(args.neighbors),
        "outputs": out_files,
        "flip_reports": flip_reports,
        "note_metric": "Leaderboard Accuracy (we generate candidates; you report the best LB)."
    }
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"[INFO] Saved summary -> {args.out_prefix}_summary.json")

if __name__ == "__main__":
    main()
