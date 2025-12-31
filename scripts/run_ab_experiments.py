#!/usr/bin/env python3
"""Run quick A/B experiments for fallback and stability filtering on top candidates.

Produces a CSV with per-candidate, per-mode metrics.
"""
from pathlib import Path
import json
import argparse
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Small mapping from model name to a constructor we can use for quick experiments
MODEL_MAP = {
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'CatBoostClassifier': None,  # may not be installed in env
    'XGBClassifier': None,
    'LGBMClassifier': None
}

# Ensure project src is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_coordinator import create_default_coordinator


def parse_params(s: str):
    try:
        return json.loads(s)
    except Exception:
        import ast
        return ast.literal_eval(s)


def evaluate_candidate(coord, X_train_full, y_train_full, X_test, y_test, cand_params, model_name=None, mode='baseline', cv=3):
    # mode: 'baseline' (no fallback), 'fallback' (allow fallback), 'stability' (stability filter), 'combined'
    smote_cfg = {"enabled": bool(cand_params.get('smote__enabled', True)),
                 "categorical_feature_names": cand_params.get('smote__categorical_feature_names', []),
                 "k_neighbors": int(cand_params.get('smote__k_neighbors', 5)),
                 "allow_fallback": False}
    # baseline: keep fallback disabled
    if mode in ('fallback', 'combined'):
        smote_cfg['allow_fallback'] = True

    # helper to run CV and collect fold info
    from sklearn.ensemble import RandomForestClassifier
    model_ctor = MODEL_MAP.get(model_name) if model_name else None
    # fallback to RandomForest which provides feature_importances_ for feature selection
    base_est = clone(model_ctor()) if (model_ctor is not None) else clone(RandomForestClassifier())

    def run_cv(X, y, sm_cfg):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        fold_fs = []
        fold_f1s = []
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            pipeline = coord.create_pipeline(base_estimator=clone(base_est), smote_config=sm_cfg, feature_selection_config={'max_features': cand_params.get('feature_selecting_classifier__max_features', None), 'threshold': cand_params.get('feature_selecting_classifier__threshold', None)})
            try:
                pipeline.set_params(**cand_params)
            except Exception:
                pass
            fitted = coord.fit_pipeline(pipeline, X_tr, y_tr)
            # inspect feature selector
            sel = getattr(fitted.named_steps.get('feature_selecting_classifier'), 'selected_features_', None)
            fold_fs.append(list(sel) if sel is not None else [])
            y_pred = fitted.predict(X_val)
            from sklearn.metrics import f1_score
            fold_f1s.append(float(f1_score(y_val, y_pred, average='macro')))
        return fold_fs, fold_f1s

    # Mode handling
    if mode in ('baseline', 'fallback'):
        fold_fs, fold_f1s = run_cv(X_train_full, y_train_full, smote_cfg)
        mean_f1 = float(np.mean(fold_f1s))
        # fit full pipeline and test

        model_ctor = MODEL_MAP.get(model_name) if model_name else None
        base_est_full = clone(model_ctor()) if (model_ctor is not None) else clone(RandomForestClassifier())
        pipeline_full = coord.create_pipeline(base_estimator=base_est_full, smote_config=smote_cfg, feature_selection_config={'max_features': cand_params.get('feature_selecting_classifier__max_features', None), 'threshold': cand_params.get('feature_selecting_classifier__threshold', None)})
        try:
            pipeline_full.set_params(**cand_params)
        except Exception:
            pass
        pipeline_full.fit(X_train_full, y_train_full)
        y_pred_test = pipeline_full.predict(X_test)
        test_f1 = float(f1_score(y_test, y_pred_test, average='macro'))
        return {'mean_f1': mean_f1, 'test_f1': test_f1, 'fold_std': float(np.std(fold_f1s)), 'selected_features': fold_fs}

    # Stability modes: first compute stable features from baseline CV
    fold_fs, fold_f1s = run_cv(X_train_full, y_train_full, {**smote_cfg, 'allow_fallback': False})
    # frequency
    from collections import Counter
    cnt = Counter()
    for fl in fold_fs:
        for f in fl:
            cnt[str(f)] += 1
    nfolds = len(fold_fs)
    stable_features = [k for k, v in cnt.items() if (v / nfolds) >= 0.6]

    # Re-evaluate restricting to stable_features
    if len(stable_features) == 0:
        # no stable features, return baseline summary
        return {'mean_f1': float(np.mean(fold_f1s)), 'test_f1': None, 'fold_std': float(np.std(fold_f1s)), 'selected_features': fold_fs, 'stable_features': stable_features}

    Xtr_sub = X_train_full.loc[:, [c for c in stable_features if c in X_train_full.columns]]
    Xte_sub = X_test.loc[:, [c for c in stable_features if c in X_test.columns]]
    fold_fs2, fold_f1s2 = run_cv(Xtr_sub, y_train_full, {**smote_cfg, 'allow_fallback': mode == 'combined'})
    from sklearn.ensemble import RandomForestClassifier
    model_ctor = MODEL_MAP.get(model_name) if model_name else None
    base_est_full = clone(model_ctor()) if (model_ctor is not None) else clone(RandomForestClassifier())
    pipeline_full = coord.create_pipeline(base_estimator=base_est_full, smote_config={**smote_cfg, 'allow_fallback': mode == 'combined'}, feature_selection_config={'max_features': None, 'threshold': None})
    try:
        pipeline_full.set_params(**cand_params)
    except Exception:
        pass
    pipeline_full.fit(Xtr_sub, y_train_full)
    y_pred_test = pipeline_full.predict(Xte_sub)
    test_f1 = float(f1_score(y_test, y_pred_test, average='macro'))
    return {'mean_f1': float(np.mean(fold_f1s2)), 'test_f1': test_f1, 'fold_std': float(np.std(fold_f1s2)), 'selected_features': fold_fs2, 'stable_features': stable_features}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--data-csv', type=Path, default=Path('data/prefi_weaviate_clean-1_flattened.csv'))
    parser.add_argument('--train-csv', type=Path, default=None, help='Optional precomputed train CSV to use instead of internal holdout split')
    parser.add_argument('--test-csv', type=Path, default=None, help='Optional precomputed test CSV to use instead of internal holdout split')
    parser.add_argument('--results-dir', type=Path, default=Path('models/run_ab_experiments'))
    args = parser.parse_args()

    outdir = args.results_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv('models/run_50_samples/search_results.csv')
    top = df.sort_values('mean_f1', ascending=False).head(args.top_k)

    coord = create_default_coordinator()

    # load full data
    df = pd.read_csv(args.data_csv)
    # mimic run_search preprocessing
    from src.utils import sanitize_column_name
    from src.ingest import _load_golden_schema, _parse_schema, preprocess_dataframe
    df.columns = [sanitize_column_name(c) for c in df.columns]
    schema = _load_golden_schema(Path('src/column_headers.json'))
    sorted_schema, column_map = _parse_schema(schema)
    df = preprocess_dataframe(df, sorted_schema, column_map)

    headers = None
    try:
        from src.utils import load_column_headers
        headers = load_column_headers(Path('src/column_headers.json'), df)
    except Exception:
        headers = {'feature_cols': [c for c in df.columns if c != 'target'], 'target_cols': ['target']}

    feature_cols = [c for c in headers.get('feature_cols', []) if c in df.columns]
    target_cols = [c for c in headers.get('target_cols', []) if c in df.columns]
    if not target_cols:
        raise ValueError('No target columns present')
    status_target = next((t for t in target_cols if t.lower().endswith('status_label')), target_cols[0])

    X = df[feature_cols].copy()
    y = df[status_target].astype(int)

    # Use explicit train/test CSVs if provided (e.g., produced by src/allocate.py)
    if args.train_csv and args.test_csv:
        df_tr = pd.read_csv(args.train_csv)
        df_te = pd.read_csv(args.test_csv)
        # sanitize and preprocess like above
        df_tr.columns = [sanitize_column_name(c) for c in df_tr.columns]
        df_te.columns = [sanitize_column_name(c) for c in df_te.columns]
        df_tr = preprocess_dataframe(df_tr, sorted_schema, column_map)
        df_te = preprocess_dataframe(df_te, sorted_schema, column_map)
        X_train_full = df_tr[feature_cols].copy().reset_index(drop=True)
        y_train_full = df_tr[status_target].astype(int).to_numpy()
        X_test = df_te[feature_cols].copy().reset_index(drop=True)
        y_test = df_te[status_target].astype(int).to_numpy()
    else:
        # holdout split
        n = len(X)
        test_n = max(100, int(0.2 * n))
        X_train_full = X.iloc[:-test_n].reset_index(drop=True)
        y_train_full = y.iloc[:-test_n].to_numpy()
        X_test = X.iloc[-test_n:].reset_index(drop=True)
        y_test = y.iloc[-test_n:].to_numpy()

    records = []
    for idx, row in top.iterrows():
        cand = parse_params(row['params'])
        for mode in ('baseline', 'fallback', 'stability', 'combined'):
            res = evaluate_candidate(coord, X_train_full, y_train_full, X_test, y_test, cand, model_name=row['model'], mode=mode, cv=3)
            records.append({'model': row['model'], 'params': row['params'], 'mode': mode, 'mean_f1': res.get('mean_f1'), 'test_f1': res.get('test_f1'), 'fold_std': res.get('fold_std'), 'stable_features': json.dumps(res.get('stable_features', []))})
            pd.DataFrame(records).to_csv(outdir / 'ab_results.csv', index=False)


if __name__ == '__main__':
    main()
