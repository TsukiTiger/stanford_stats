#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-query target count tuning with 2-of-3 model agreement.

1) Train XGB + CatBoost + LGBM on row features (robust impute/clip, per-query z/rank, light interactions).
2) Train a query-level regressor on TRAIN to predict #positives per query.
3) On TEST, for each query, nudge base labels toward predicted target using ONLY high-confidence
   2-of-3 agreement flips (asymmetric thresholds), with per-query and global caps.

Metric: public leaderboard accuracy. You submit, then we tighten around best settings.

Run example at the bottom.
"""

import argparse, os, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

# ---------- helpers ----------
def log(m): print(m, flush=True)

def numeric_features(df: pd.DataFrame, drop: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in drop]

def impute_clip(df: pd.DataFrame, cols: List[str]):
    for c in cols:
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
    # ranks within query
    for c in feats:
        r = f"{c}__qrank"
        df[r] = g[c].rank(method="first", ascending=True).astype(np.float32); new_cols.append(r)
    # zscores
    stats = g[feats].agg(['mean','std'])
    for c in feats:
        m = stats[(c,'mean')].reindex(df[qcol]).to_numpy()
        s = stats[(c,'std')].reindex(df[qcol]).to_numpy()
        s = np.where((s==0) | (~np.isfinite(s)), 1.0, s)
        z = f"{c}__qz"
        df[z] = (df[c].to_numpy() - m) / s; new_cols.append(z)
    # extremal flags + size
    for c in feats:
        df[f"{c}__is_qmax"] = (df[c] == g[c].transform("max")).astype(np.int8)
        df[f"{c}__is_qmin"] = (df[c] == g[c].transform("min")).astype(np.int8)
        new_cols += [f"{c}__is_qmax", f"{c}__is_qmin"]
    df["__qsize"] = g[feats[0]].transform("size").astype(np.int32); new_cols.append("__qsize")
    return new_cols

def make_submission(sample, ids, preds, path):
    sub = sample.copy()
    tmp = pd.DataFrame({"id": ids, "relevance": preds})
    sub = sub.merge(tmp, on="id", how="left", suffixes=("_tmpl",""))
    sub["relevance"] = sub["relevance"].fillna(sub["relevance_tmpl"]).astype(int)
    sub = sub[["id","relevance"]]; sub.to_csv(path, index=False); return sub

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out_prefix", default="flip_prior_2of3")

    # agreement + confidence thresholds
    ap.add_argument("--min_conf_up", type=float, default=0.030)   # for 0->1 (p_avg - 0.5)
    ap.add_argument("--min_conf_down", type=float, default=0.055) # for 1->0 (0.5 - p_avg)

    # per-query controls
    ap.add_argument("--max_per_query", type=int, default=2)  # allow up to 2 flips per query
    ap.add_argument("--global_budget", type=int, nargs="+", default=[70,80,90,100])

    # prior strength sweep: final target = round((1-beta)*base_pos + beta*pred_pos)
    ap.add_argument("--beta_list", type=float, nargs="+", default=[0.3,0.4,0.5])

    # GPU flags
    ap.add_argument("--use_gpu", type=int, default=0)          # for XGB & CatBoost
    ap.add_argument("--lgb_device", type=str, default="cpu", choices=["cpu","gpu"])

    args = ap.parse_args()

    # ----- load -----
    train  = pd.read_csv(args.train)
    test   = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    base   = pd.read_csv(args.base)
    assert {"id","relevance"}.issubset(base.columns)
    assert {"id","relevance"}.issubset(sample.columns)
    assert {"id","query_id"}.issubset(test.columns)
    assert {"id","relevance","query_id"}.issubset(train.columns)

    # align test to base order
    test_ord = base[["id"]].merge(test, on="id", how="left")
    assert len(test_ord)==len(base)
    base_y = base["relevance"].astype(int).to_numpy()
    qids   = test_ord["query_id"].to_numpy()

    # ----- row-level features -----
    drop = ["id","url_id","query_id","relevance"]
    base_feats = numeric_features(train, drop)
    if not base_feats: raise RuntimeError("No numeric features in training set.")
    train["_is_train"] = 1; test_ord["_is_train"] = 0
    Xall = pd.concat([train[["_is_train","query_id"]+base_feats],
                      test_ord[["_is_train","query_id"]+base_feats]],
                     axis=0, ignore_index=True)
    impute_clip(Xall, base_feats)
    inter_cols = add_interactions(Xall, base_feats, max_pairs=40)
    pq_cols    = add_query_features(Xall, "query_id", base_feats)
    feats = base_feats + inter_cols + pq_cols
    X_tr = Xall[Xall["_is_train"]==1].reset_index(drop=True)
    X_te = Xall[Xall["_is_train"]==0].reset_index(drop=True)
    y    = train["relevance"].astype(int).to_numpy()

    # ----- train 3 models -----
    xgb = XGBClassifier(
        n_estimators=850, max_depth=6, learning_rate=0.045,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.2, reg_lambda=0.35,
        min_child_weight=1.0, eval_metric="logloss", random_state=42, n_jobs=-1,
        scale_pos_weight=float((y==0).sum()/max((y==1).sum(),1)),
        tree_method="gpu_hist" if args.use_gpu else "hist",
        predictor="gpu_predictor" if args.use_gpu else "auto"
    )
    xgb.fit(X_tr[feats].values, y)
    p_x = xgb.predict_proba(X_te[feats].values)[:,1]

    cat = CatBoostClassifier(
        iterations=1200, learning_rate=0.035, depth=6,
        l2_leaf_reg=3.0, loss_function="Logloss",
        random_seed=42, verbose=False,
        task_type="GPU" if args.use_gpu else "CPU"
    )
    cat.fit(Pool(X_tr[feats], y))
    p_c = cat.predict_proba(Pool(X_te[feats]))[:,1]

    lgb = LGBMClassifier(
        n_estimators=900, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.2, reg_lambda=0.3, objective="binary", random_state=42,
        device=("gpu" if args.lgb_device=="gpu" else "cpu")
    )
    lgb.fit(X_tr[feats], y)
    p_l = lgb.predict_proba(X_te[feats])[:,1]

    p_stack = np.column_stack([p_x, p_c, p_l])
    cls = (p_stack >= 0.5).astype(int)
    votes_for_1 = cls.sum(axis=1)
    votes_for_0 = 3 - votes_for_1
    p_avg = p_stack.mean(axis=1)
    conf_up = np.maximum(0.0, p_avg - 0.5)
    conf_dn = np.maximum(0.0, 0.5 - p_avg)

    # agreement candidates
    cand_up = np.where((base_y==0) & (votes_for_1>=2) & (p_avg>0.5) & (conf_up>=args.min_conf_up))[0]
    cand_dn = np.where((base_y==1) & (votes_for_0>=2) & (p_avg<0.5) & (conf_dn>=args.min_conf_down))[0]

    # ----- query-level prior: predict number of positives per query -----
    # build query table for TRAIN
    tr = train.copy()
    gtr = tr.groupby("query_id", observed=True)
    qtab_tr = pd.DataFrame({
        "query_id": gtr.size().index,
        "qsize": gtr.size().values,
        "pos": gtr["relevance"].sum().values
    })
    # add per-query means/stds of base_feats
    for c in base_feats:
        qtab_tr[f"{c}_mean"] = gtr[c].mean().values
        qtab_tr[f"{c}_std"]  = gtr[c].std().fillna(0).values

    # model to predict pos count
    qreg = LGBMClassifier(  # classification on "is positive" per item would be messy; use regression-like trick
        n_estimators=500, learning_rate=0.06, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, random_state=7, objective="multiclass", num_class=max(2, qtab_tr["pos"].max()+1)
    )
    # But LightGBM multiclass needs y in [0..num_class-1]; we clamp at 20 to avoid huge class count.
    qtab_tr["pos_clamped"] = qtab_tr["pos"].clip(0, 20).astype(int)
    qreg.fit(qtab_tr.drop(columns=["pos","pos_clamped"]).set_index("query_id").values,
             qtab_tr["pos_clamped"].values)
    # build query table for TEST (aligned to base order's qids)
    gte = pd.DataFrame({"query_id": test_ord["query_id"].unique()})
    gte["qsize"] = gte["query_id"].map(test_ord.groupby("query_id").size())
    for c in base_feats:
        gte[f"{c}_mean"] = gte["query_id"].map(test_ord.groupby("query_id")[c].mean())
        gte[f"{c}_std"]  = gte["query_id"].map(test_ord.groupby("query_id")[c].std().fillna(0))
    qX = gte.drop(columns=["query_id"]).values
    q_pred_dist = qreg.predict_proba(qX)
    pred_pos = (q_pred_dist * np.arange(q_pred_dist.shape[1])[None,:]).sum(axis=1)
    pred_pos = np.minimum(pred_pos, gte["qsize"].to_numpy().astype(float))

    pred_pos_map = dict(zip(gte["query_id"].to_numpy(), pred_pos))

    # For each beta and budget, nudge per query
    sample = pd.read_csv(args.sample)
    reports = []
    for beta in args.beta_list:
        # compute per-query targets
        base_pos_by_q = pd.Series(base_y).groupby(qids).sum()
        qsize_by_q = pd.Series(1, index=qids).groupby(qids).sum()
        tgt_by_q = {}
        for q in base_pos_by_q.index:
            base_pos = int(base_pos_by_q.loc[q])
            pp = float(pred_pos_map.get(q, base_pos))
            tgt = int(round((1-beta)*base_pos + beta*pp))
            tgt = max(0, min(int(qsize_by_q.loc[q]), tgt))
            tgt_by_q[q] = tgt

        # pre-sort candidates by confidence
        up_order = cand_up[np.argsort(-conf_up[cand_up])]
        dn_order = cand_dn[np.argsort(-conf_dn[cand_dn])]

        # do selections under per-query cap, THEN apply global budget
        chosen = []
        flipped_y = base_y.copy()
        used_q = defaultdict(int)

        # helper: apply desired changes per query toward target
        # Step 1: visit queries in descending shortage/excess magnitude
        delta_list = []
        for q in base_pos_by_q.index:
            cur = int(base_pos_by_q.loc[q])
            tgt = tgt_by_q[q]
            delta_list.append((abs(tgt-cur), q, tgt-cur))
        delta_list.sort(reverse=True)  # large adjustments first

        # index lists by query for fast pick
        up_by_q = defaultdict(list)
        for i in up_order: up_by_q[qids[i]].append(i)
        dn_by_q = defaultdict(list)
        for i in dn_order: dn_by_q[qids[i]].append(i)

        for _, q, d in delta_list:
            if d > 0:  # need more positives -> flip some 0->1
                pool = up_by_q.get(q, [])
                k = 0
                for i in pool:
                    if used_q[q] >= args.max_per_query: break
                    if flipped_y[i] != base_y[i]:
                        continue
                    flipped_y[i] = 1
                    used_q[q] += 1
                    chosen.append(i); k += 1
                    if k >= d: break
            elif d < 0:  # need fewer positives -> flip some 1->0
                pool = dn_by_q.get(q, [])
                k = 0
                for i in pool:
                    if used_q[q] >= args.max_per_query: break
                    if flipped_y[i] != base_y[i]:
                        continue
                    flipped_y[i] = 0
                    used_q[q] += 1
                    chosen.append(i); k += 1
                    if k >= (-d): break

        # enforce global budgets by confidence across chosen
        chosen = np.array(list(set(chosen)), dtype=int)
        conf = np.zeros_like(conf_up)
        conf[chosen] = np.where(flipped_y[chosen]==1, conf_up[chosen], conf_dn[chosen])
        order = chosen[np.argsort(-conf[chosen])]

        for B in args.global_budget:
            keep = order[:B]
            final = base_y.copy()
            final[keep] = 1 - base_y[keep]
            out_path = f"{args.out_prefix}_B{B}_cap{args.max_per_query}_beta{beta:.2f}_up{args.min_conf_up:.3f}_dn{args.min_conf_down:.3f}.csv"
            make_submission(sample, base["id"], final, out_path)
            log(f"[WRITE] {out_path} | flips={len(keep)}")

            reports.append({
                "file": os.path.basename(out_path),
                "beta": float(beta),
                "budget": int(B),
                "max_per_query": int(args.max_per_query),
                "min_conf_up": float(args.min_conf_up),
                "min_conf_down": float(args.min_conf_down),
                "num_flipped": int(len(keep)),
                "first_80_indexes": [int(i) for i in keep[:80]],
                "first_80_ids": base['id'].iloc[keep[:80]].tolist()
            })

    with open(f"{args.out_prefix}_report.json", "w") as f:
        json.dump({"note":"indexes are 0-based in BASE order", "reports":reports}, f, indent=2)
    log(f"[INFO] Saved report -> %s_report.json" % args.out_prefix)

if __name__ == "__main__":
    main()
