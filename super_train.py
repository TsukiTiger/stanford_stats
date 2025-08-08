import os, sys, warnings, argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Optional CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# Optional Optuna
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

def read_data(train_path, test_path):
    print(f"[INFO] Reading {train_path} / {test_path}")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    target = "relevance"
    group  = "query_id"
    drop_cols = ["id", "url_id", group, target]
    features = [c for c in train.columns if c not in drop_cols]
    X = train[features].copy()
    y = train[target].astype(int).values
    g = train[group].values
    X_test = test[features].copy()

    print(f"[INFO] Train shape: {train.shape} | Test shape: {test.shape} | n_features: {len(features)}")
    return train, test, X, y, g, X_test, features

def add_smart_interactions(X, y, X_test, feature_names, top_k=6, do_sumdiff=True, do_prodratio=True):
    """Pick top_k features via MI+corr and create limited interactions."""
    df = X.copy()
    df_test = X_test.copy()

    mi = mutual_info_classif(df[feature_names], y, discrete_features=False, random_state=0)
    mi_rank = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    corr_vals = {}
    for f in feature_names:
        v = df[f].values
        if np.std(v) == 0:
            corr_vals[f] = 0.0
        else:
            corr_vals[f] = abs(np.corrcoef(v, y)[0, 1])
    corr = pd.Series(corr_vals).sort_values(ascending=False)

    score = (mi_rank.rank(ascending=False) + corr.rank(ascending=False)).sort_values()
    keep = list(score.index[:min(top_k, len(feature_names))])

    def safe_div(a, b): return a / (b + 1e-6)

    new_cols = []
    for i in range(len(keep)):
        for j in range(i+1, len(keep)):
            f1, f2 = keep[i], keep[j]
            if do_prodratio:
                df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
                df_test[f"{f1}_x_{f2}"] = df_test[f1] * df_test[f2]
                new_cols.append(f"{f1}_x_{f2}")

                df[f"{f1}_div_{f2}"] = safe_div(df[f1], df[f2])
                df_test[f"{f1}_div_{f2}"] = safe_div(df_test[f1], df_test[f2])
                new_cols.append(f"{f1}_div_{f2}")

                df[f"{f2}_div_{f1}"] = safe_div(df[f2], df[f1])
                df_test[f"{f2}_div_{f1}"] = safe_div(df_test[f2], df_test[f1])
                new_cols.append(f"{f2}_div_{f1}")

            if do_sumdiff:
                df[f"{f1}_plus_{f2}"] = df[f1] + df[f2]
                df_test[f"{f1}_plus_{f2}"] = df_test[f1] + df_test[f2]
                new_cols.append(f"{f1}_plus_{f2}")

                df[f"{f1}_minus_{f2}"] = df[f1] - df[f2]
                df_test[f"{f1}_minus_{f2}"] = df_test[f1] - df_test[f2]
                new_cols.append(f"{f1}_minus_{f2}")

                df[f"{f2}_minus_{f1}"] = df[f2] - df[f1]
                df_test[f"{f2}_minus_{f1}"] = df_test[f2] - df_test[f1]
                new_cols.append(f"{f2}_minus_{f1}")

    print(f"[INFO] Added {len(new_cols)} interactions among top {len(keep)} features: {keep}")
    return df, df_test

def model_configs(y, seed, lgb_estimators=600, xgb_estimators=600):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    spw = max(neg / max(pos, 1), 0.5)

    lgbm = LGBMClassifier(
        n_estimators=lgb_estimators,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.2,
        reg_lambda=0.2,
        objective="binary",
        class_weight="balanced",
        random_state=seed,
    )
    xgb = XGBClassifier(
        n_estimators=xgb_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        tree_method="hist",
    )
    cat = None
    if HAS_CAT:
        cat = CatBoostClassifier(
            iterations=lgb_estimators,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed,
            verbose=False,
            scale_pos_weight=spw,
        )
    return lgbm, xgb, cat

def try_early_stopping(clf, X_tr, y_tr, X_va, y_va, lib="lgbm"):
    try:
        if lib == "xgb":
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=100)
        elif lib == "cat":
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="logloss", verbose=False, early_stopping_rounds=100)
        return clf
    except TypeError:
        clf.fit(X_tr, y_tr)
        return clf

def optimize_blend_weights(oof_list, y, refine=1):
    """Coarse->fine weight search + threshold sweep."""
    m = len(oof_list)
    if m == 1:
        p = oof_list[0]
        best_th, best_acc = 0.5, 0.0
        for t in np.linspace(0.35, 0.65, 61):
            a = accuracy_score(y, (p >= t).astype(int))
            if a > best_acc:
                best_acc, best_th = a, t
        return np.array([1.0]), best_acc, best_th

    oof_stack = np.vstack(oof_list).T
    ths = np.linspace(0.35, 0.65, 61)
    best_w = np.ones(m)/m
    best_acc, best_th = -1.0, 0.5

    def eval_w(w):
        p = oof_stack @ w
        acc, th = -1, 0.5
        for t in ths:
            a = accuracy_score(y, (p >= t).astype(int))
            if a > acc:
                acc, th = a, t
        return acc, th

    # coarse grid (m=2/3) else random
    grid = np.linspace(0, 1, 11)
    if m == 2:
        for w0 in grid:
            w = np.array([w0, 1 - w0])
            acc, th = eval_w(w)
            if acc > best_acc:
                best_acc, best_th, best_w = acc, th, w
    elif m == 3:
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 <= 1:
                    w = np.array([w0, w1, 1 - w0 - w1])
                    acc, th = eval_w(w)
                    if acc > best_acc:
                        best_acc, best_th, best_w = acc, th, w
    else:
        for _ in range(500):
            w = np.random.dirichlet(np.ones(m))
            acc, th = eval_w(w)
            if acc > best_acc:
                best_acc, best_th, best_w = acc, th, w

    # refine by jitter
    for _ in range(max(refine, 0)):
        for __ in range(300):
            cand = best_w + np.random.normal(0, 0.15, size=m)
            cand = np.clip(cand, 0, 1)
            s = cand.sum()
            if s == 0: 
                continue
            cand /= s
            acc, th = eval_w(cand)
            if acc > best_acc:
                best_acc, best_th, best_w = acc, th, cand

    return best_w, best_acc, best_th

def maybe_optuna_tune(X, y, g, seeds, trials):
    if not HAS_OPTUNA or trials <= 0:
        return None
    print(f"[INFO] Starting Optuna (trials={trials})…")
    def objective(trial):
        leaves = trial.suggest_int("num_leaves", 15, 63)
        lr    = trial.suggest_float("lgb_lr", 0.02, 0.1, log=True)
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seeds[0])
        accs = []
        for trn, val in sgkf.split(X, y, g):
            clf = LGBMClassifier(
                n_estimators=400, learning_rate=lr, num_leaves=leaves,
                feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
                reg_alpha=0.2, reg_lambda=0.2, objective="binary",
                class_weight="balanced", random_state=seeds[0]
            )
            clf.fit(X.iloc[trn], y[trn])
            p = clf.predict_proba(X.iloc[val])[:,1]
            accs.append(accuracy_score(y[val], (p>=0.5).astype(int)))
        return float(np.mean(accs))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    print("[INFO] Optuna best:", study.best_params, "value:", study.best_value)
    return study.best_params

def run(args):
    train_df, test_df, X0, y, groups, X0_test, base_feats = read_data(args.train, args.test)

    # FE: smart interactions
    X, X_test = add_smart_interactions(X0, y, X0_test, base_feats,
                                       top_k=args.topk, do_sumdiff=True, do_prodratio=True)

    # Preprocess (passthrough kept for future edits)
    pre = ColumnTransformer([("num", "passthrough", list(X.columns))], remainder="drop")

    # seeds
    seeds = args.seeds if len(args.seeds) else [args.seed]
    print(f"[INFO] Using seeds: {seeds}")

    # Optional: quick hyperparam warm start
    warm = maybe_optuna_tune(X, y, groups, seeds, args.trials)

    # Collect per-seed oof/test for later averaging
    oof_all, test_all, per_seed_summaries = [], [], []

    for seed in seeds:
        print(f"\n===================== SEED {seed} =====================")
        lgbm, xgb, cat = model_configs(y, seed, args.lgb_estimators, args.xgb_estimators)
        if warm and "num_leaves" in warm:
            try: lgbm.set_params(num_leaves=warm["num_leaves"])
            except: pass

        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)

        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cat = np.zeros(len(X)) if HAS_CAT and args.use_cat else None

        test_lgb = np.zeros(len(X_test))
        test_xgb = np.zeros(len(X_test))
        test_cat = np.zeros(len(X_test)) if HAS_CAT and args.use_cat else None

        for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
            X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
            y_tr, y_va = y[trn_idx], y[val_idx]

            X_tr_p = pre.fit_transform(X_tr)
            X_va_p = pre.transform(X_va)
            X_te_p = pre.transform(X_test)

            # LGB
            lgb_f = try_early_stopping(lgbm, X_tr_p, y_tr, X_va_p, y_va, lib="lgbm")
            p_lgb_va = lgb_f.predict_proba(X_va_p)[:,1]
            p_lgb_te = lgb_f.predict_proba(X_te_p)[:,1]
            oof_lgb[val_idx] = p_lgb_va
            test_lgb += p_lgb_te / args.folds

            # XGB
            xgb_f = try_early_stopping(xgb, X_tr_p, y_tr, X_va_p, y_va, lib="xgb")
            p_xgb_va = xgb_f.predict_proba(X_va_p)[:,1]
            p_xgb_te = xgb_f.predict_proba(X_te_p)[:,1]
            oof_xgb[val_idx] = p_xgb_va
            test_xgb += p_xgb_te / args.folds

            # CatBoost
            if HAS_CAT and args.use_cat:
                cat_f = try_early_stopping(cat, X_tr_p, y_tr, X_va_p, y_va, lib="cat")
                p_cat_va = cat_f.predict_proba(X_va_p)[:,1]
                p_cat_te = cat_f.predict_proba(X_te_p)[:,1]
                oof_cat[val_idx] = p_cat_va
                test_cat += p_cat_te / args.folds

            blend_va = 0.5 * p_lgb_va + 0.5 * p_xgb_va
            acc_05 = accuracy_score(y_va, (blend_va >= 0.5).astype(int))
            print(f"[Seed {seed} | Fold {fold}] LGB mean={p_lgb_va.mean():.4f} | XGB mean={p_xgb_va.mean():.4f} | Blend@0.5 acc={acc_05:.4f}")

        base_oofs = [oof_lgb, oof_xgb]
        base_tests = [test_lgb, test_xgb]
        model_names = ["lgb", "xgb"]
        if HAS_CAT and args.use_cat and oof_cat is not None:
            base_oofs.append(oof_cat)
            base_tests.append(test_cat)
            model_names.append("cat")

        # Meta LR
        from sklearn.pipeline import Pipeline
        meta_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000))
        ])
        stack_train = np.vstack(base_oofs).T
        stack_test  = np.vstack(base_tests).T
        meta_lr.fit(stack_train, y)
        oof_meta_lr = meta_lr.predict_proba(stack_train)[:,1]
        test_meta_lr = meta_lr.predict_proba(stack_test)[:,1]

        # Meta XGB (optional)
        if args.stack_meta == "xgb":
            meta_x = XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.07,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=seed
            )
            meta_x.fit(stack_train, y)
            oof_meta_x = meta_x.predict_proba(stack_train)[:,1]
            test_meta_x = meta_x.predict_proba(stack_test)[:,1]
            oofs_to_blend = base_oofs + [oof_meta_lr, oof_meta_x]
            tests_to_blend = base_tests + [test_meta_lr, test_meta_x]
            model_names += ["meta_lr", "meta_x"]
        else:
            oofs_to_blend = base_oofs + [oof_meta_lr]
            tests_to_blend = base_tests + [test_meta_lr]
            model_names += ["meta_lr"]

        w, acc, th = optimize_blend_weights(oofs_to_blend, y, refine=args.blend_refine)
        pretty_w = {k: float(v) for k, v in zip(model_names[:len(w)], w)}
        print(f"[Seed {seed}] Best blend OOF acc={acc:.5f} @ thr={th:.4f} with weights {pretty_w}")

        oof_all.append(np.vstack(oofs_to_blend))
        test_all.append(np.vstack(tests_to_blend))

    # Average probs across seeds per model
    M = oof_all[0].shape[0]
    oof_avg_models, test_avg_models = [], []
    for m in range(M):
        oof_avg_models.append(np.mean([oof_all[s][m] for s in range(len(oof_all))], axis=0))
        test_avg_models.append(np.mean([test_all[s][m] for s in range(len(test_all))], axis=0))

    final_w, final_acc, final_th = optimize_blend_weights(oof_avg_models, y, refine=args.blend_refine)
    print(f"\n[FINAL] Best OOF acc={final_acc:.5f} @ thr={final_th:.4f}")
    print(f"[FINAL] Weights: {np.round(final_w,4).tolist()} (aligned to order used above)")

    # Sanity check
    final_oof_prob = np.vstack(oof_avg_models).T @ final_w
    print(f"[CHECK] OOF accuracy @ final thr: {accuracy_score(y, (final_oof_prob >= final_th).astype(int)):.5f}")

    # Final test preds
    final_test_prob = np.vstack(test_avg_models).T @ final_w
    test_preds = (final_test_prob >= final_th).astype(int)

    # ==== ID handling (this fixes your grader issue) ====
    if args.sample:
        assert os.path.exists(args.sample), f"Sample file not found: {args.sample}"
        sample_df = pd.read_csv(args.sample)
        assert len(sample_df) == len(test_preds), f"Sample rows ({len(sample_df)}) != test preds ({len(test_preds)})"
        sub = sample_df.copy()
        # Detect relevance column name in sample; set to 'relevance'
        rel_col = "relevance" if "relevance" in sub.columns else "Relevance" if "Relevance" in sub.columns else None
        if rel_col is None:
            sub["relevance"] = test_preds
        else:
            sub[rel_col] = test_preds
            if rel_col != "relevance":
                sub = sub.rename(columns={rel_col: "relevance"})
    else:
        # Fallback: if test has 'id' column, reuse
        if "id" in test_df.columns:
            sub = test_df[["id"]].copy()
            sub["relevance"] = test_preds
        else:
            raise ValueError("No --sample provided and test.csv has no 'id' column. Pass --sample path.")

    sub.to_csv(args.out, index=False)
    print(f"[INFO] Wrote submission to {args.out} | rows={len(sub)}")
    print(sub.head(10).to_string(index=False))

    summary = {
        "final_acc": float(final_acc),
        "final_thr": float(final_th),
        "weights": list(map(float, final_w)),
        "n_features_after_FE": int(X.shape[1]),
        "seeds": list(map(int, args.seeds if len(args.seeds) else [args.seed])),
        "used_catboost": bool(HAS_CAT and args.use_cat),
        "stack_meta": args.stack_meta,
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary -> {os.path.splitext(args.out)[0] + '_summary.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="training.csv")
    p.add_argument("--test",  default="test.csv")
    p.add_argument("--sample", default="", help="Path to sample_submission.csv (to copy ID order)")
    p.add_argument("--out",   default="submission.csv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    p.add_argument("--lgb_estimators", type=int, default=600)
    p.add_argument("--xgb_estimators", type=int, default=600)
    p.add_argument("--use_cat", action="store_true", help="Include CatBoost as base model (if installed)")
    p.add_argument("--stack_meta", choices=["lr","xgb"], default="lr", help="Meta-learner for stacking")
    p.add_argument("--topk", type=int, default=6, help="Top-K base features to interact")
    p.add_argument("--blend_refine", type=int, default=1, help="Refinement rounds in blend search")
    p.add_argument("--trials", type=int, default=0, help="Optuna trials for quick warm-start")
    args = p.parse_args()
    run(args)
