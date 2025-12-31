#!/usr/bin/env python3
"""Run diagnostics for CV->test gap, ablations, and resampling strategies.

Produces CSV/JSON summaries and plots under the results dir.
"""
from pathlib import Path
import json
import argparse
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_coordinator import create_default_coordinator
from src.components.feature_selector import FeatureSelectingClassifier
from src.components.smote_sampler import MaybeSMOTESampler

# samplers we'll try
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Basic model map
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
MODEL_MAP = {
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
}


def parse_params(s: str):
    try:
        return json.loads(s)
    except Exception:
        import ast
        return ast.literal_eval(s)


def ks_numeric_pairs(Xa: pd.DataFrame, Xb: pd.DataFrame, numeric_cols):
    from scipy.stats import ks_2samp
    out = {}
    for c in numeric_cols:
        try:
            s = ks_2samp(Xa[c].dropna(), Xb[c].dropna()).statistic
        except Exception:
            s = np.nan
        out[c] = float(s) if not np.isnan(s) else None
    return out


def run_diagnostics_for_candidate(coord, X_train_full, y_train_full, X_test, y_test, cand, model_name, outdir, cv=3):
    model_ctor = MODEL_MAP.get(model_name, RandomForestClassifier)
    base_est = clone(model_ctor())

    # baseline CV (no fallback)
    sm_cfg = {"enabled": bool(cand.get('smote__enabled', True)),
              "categorical_feature_names": cand.get('smote__categorical_feature_names', []),
              "k_neighbors": int(cand.get('smote__k_neighbors', 5)),
              "allow_fallback": False}

    skf = __import__('sklearn').model_selection.StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_records = []
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_tr, X_val = X_train_full.iloc[tr_idx].reset_index(drop=True), X_train_full.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]

        pipeline = coord.create_pipeline(base_estimator=clone(base_est), smote_config=sm_cfg, feature_selection_config={'max_features': cand.get('feature_selecting_classifier__max_features', None), 'threshold': cand.get('feature_selecting_classifier__threshold', None)})

        # run sampler FIT_RESAMPLE explicitly for diagnostics
        sampler = pipeline.named_steps['smote']
        try:
            X_res, y_res = sampler.fit_resample(X_tr.copy(), y_tr.copy())
            sampler_used = getattr(sampler, 'last_used_sampler', None)
            fallback_used = getattr(sampler, 'last_fallback_used', None)
            min_class = getattr(sampler, 'last_min_class_count', None)
        except Exception as e:
            X_res, y_res = X_tr.copy(), y_tr.copy()
            sampler_used = str(type(sampler))
            fallback_used = None
            min_class = None

        # fit full pipeline (will run sampler internally again but that's OK)
        fitted = coord.fit_pipeline(deepcopy(pipeline), X_tr.copy(), y_tr.copy())
        sel = getattr(fitted.named_steps.get('feature_selecting_classifier'), 'selected_features_', None)
        y_pred = fitted.predict(X_val)
        fold_f1 = float(f1_score(y_val, y_pred, average='macro'))

        fold_records.append({'fold': int(fold_i), 'fold_f1': fold_f1, 'train_n': int(len(X_tr)), 'resampled_n': int(len(X_res)), 'min_class_before': int(np.bincount(y_tr).min()), 'min_class_after': int(np.bincount(y_res).min()), 'sampler_used': sampler_used, 'fallback_used': bool(fallback_used), 'selected_features': list(sel) if sel is not None else []})

    # summary stats
    df_f = pd.DataFrame(fold_records)
    df_f.to_csv(outdir / 'cv_fold_diagnostics.csv', index=False)

    # compute stable features
    from collections import Counter
    cnt = Counter()
    for fl in df_f['selected_features']:
        for f in fl:
            cnt[str(f)] += 1
    nfolds = len(df_f)
    stable_features = [k for k, v in cnt.items() if (v / nfolds) >= 0.6]

    # ablation: stable only vs non-stable
    ablation = {}
    stable = [c for c in stable_features if c in X_train_full.columns]
    nonstable = [c for c in X_train_full.columns if c not in stable]

    def fit_and_test(Xtr, ytr, Xte, yte, desc):
        if Xtr.shape[1] == 0:
            return {'desc': desc, 'test_f1': None}
        pfull = coord.create_pipeline(base_estimator=clone(base_est), smote_config=sm_cfg, feature_selection_config={'max_features': None, 'threshold': None})
        try:
            pfull.set_params(**cand)
        except Exception:
            pass
        pfull.fit(Xtr, ytr)
        ypred = pfull.predict(Xte)
        return {'desc': desc, 'test_f1': float(f1_score(yte, ypred, average='macro'))}

    ablation['stable_only'] = fit_and_test(X_train_full.loc[:, stable], y_train_full, X_test.loc[:, stable], y_test, 'stable_only')
    ablation['non_stable_only'] = fit_and_test(X_train_full.loc[:, nonstable], y_train_full, X_test.loc[:, nonstable], y_test, 'non_stable_only')

    # distribution shift tests (KS on numeric)
    num_cols = [c for c in X_train_full.columns if pd.api.types.is_numeric_dtype(X_train_full[c])]
    ks_stats = ks_numeric_pairs(X_train_full, X_test, num_cols)

    # resampling strategies
    strategies = {
        'none': None,
        'maybe_smote_no_fallback': MaybeSMOTESampler(enabled=sm_cfg['enabled'], categorical_feature_names=sm_cfg['categorical_feature_names'], k_neighbors=sm_cfg['k_neighbors'], allow_fallback=False),
        'maybe_smote_fallback': MaybeSMOTESampler(enabled=sm_cfg['enabled'], categorical_feature_names=sm_cfg['categorical_feature_names'], k_neighbors=sm_cfg['k_neighbors'], allow_fallback=True),
        'random_undersampler': RandomUnderSampler(),
        'adasyn': ADASYN(),
        'smote': SMOTE(k_neighbors=sm_cfg['k_neighbors']),
        'borderline_smote': BorderlineSMOTE(k_neighbors=sm_cfg['k_neighbors'])
    }

    strat_results = []
    for name, sampler in strategies.items():
        fold_fs = []
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            X_tr, X_val = X_train_full.iloc[tr_idx].reset_index(drop=True), X_train_full.iloc[val_idx].reset_index(drop=True)
            y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]

            if name == 'none':
                p = coord.create_pipeline(base_estimator=clone(base_est), smote_config={'enabled': False, 'categorical_feature_names': [], 'k_neighbors': 5, 'allow_fallback': False}, feature_selection_config={'max_features': cand.get('feature_selecting_classifier__max_features', None), 'threshold': cand.get('feature_selecting_classifier__threshold', None)})
                fitted = coord.fit_pipeline(p, X_tr, y_tr)
            else:
                # build pipeline with chosen sampler
                fs = FeatureSelectingClassifier(estimator=clone(base_est), max_features=cand.get('feature_selecting_classifier__max_features', None), threshold=cand.get('feature_selecting_classifier__threshold', None))
                p = ImbPipeline([('sampler', sampler), ('feature_selecting_classifier', fs)])
                # Run fit-resample then fit
                try:
                    X_res, y_res = sampler.fit_resample(X_tr.copy(), y_tr.copy())
                except Exception:
                    X_res, y_res = X_tr.copy(), y_tr.copy()
                fs.fit(X_res, y_res)
                y_pred = fs.predict(X_val)
                fold_fs.append(float(f1_score(y_val, y_pred, average='macro')))
                continue

            y_pred = fitted.predict(X_val)
            fold_fs.append(float(f1_score(y_val, y_pred, average='macro')))
        strat_results.append({'strategy': name, 'mean_f1': float(np.mean(fold_fs)), 'std_f1': float(np.std(fold_fs))})

    # write summaries
    summary = {
        'fold_records': fold_records,
        'stable_features': stable,
        'ablation': ablation,
        'ks_stats_top_numeric': {k: v for k, v in sorted(ks_stats.items(), key=lambda it: -abs(it[1]) )[:30]},
        'resampling_strategies': strat_results
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, default=lambda o: list(o) if hasattr(o, '__iter__') else str(o), indent=2))

    # plots
    plt.figure(figsize=(6,4))
    plt.plot([r['fold'] for r in fold_records], [r['fold_f1'] for r in fold_records], marker='o')
    plt.title('Per-fold F1')
    plt.xlabel('fold')
    plt.ylabel('f1_macro')
    plt.grid(True)
    plt.savefig(outdir / 'per_fold_f1.png')
    plt.close()

    # sampler used counts
    samp_counts = pd.Series([r['sampler_used'] for r in fold_records]).value_counts()
    samp_counts.to_csv(outdir / 'sampler_counts.csv')

    # KS top numeric histogram
    ksvals = pd.Series(ks_stats)
    ksvals.sort_values(ascending=False).head(30).plot(kind='bar', figsize=(10,4))
    plt.title('Top KS statistics (train vs test numeric features)')
    plt.savefig(outdir / 'ks_top_numeric.png')
    plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--results-dir', type=Path, default=Path('models/gap_diagnostics'))
    parser.add_argument('--data-csv', type=Path, default=Path('data/prefi_weaviate_clean-1_flattened.csv'))
    args = parser.parse_args()

    outbase = args.results_dir
    outbase.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv('models/run_50_samples/search_results.csv')
    top = df.sort_values('mean_f1', ascending=False).head(args.top_k)

    coord = create_default_coordinator()

    # load data
    from src.utils import sanitize_column_name
    from src.ingest import _load_golden_schema, _parse_schema, preprocess_dataframe
    from src.utils import load_column_headers

    ddf = pd.read_csv(args.data_csv)
    ddf.columns = [sanitize_column_name(c) for c in ddf.columns]
    schema = _load_golden_schema(Path('src/column_headers.json'))
    sorted_schema, column_map = _parse_schema(schema)
    ddf = preprocess_dataframe(ddf, sorted_schema, column_map)

    headers = load_column_headers(Path('src/column_headers.json'), ddf)
    feature_cols = [c for c in headers.get('feature_cols', []) if c in ddf.columns]
    target_cols = [c for c in headers.get('target_cols', []) if c in ddf.columns]
    status_target = next((t for t in target_cols if t.lower().endswith('status_label')), target_cols[0])

    X = ddf[feature_cols].copy()
    y = ddf[status_target].astype(int).to_numpy()

    n = len(X)
    test_n = max(100, int(0.2 * n))
    X_train_full = X.iloc[:-test_n].reset_index(drop=True)
    y_train_full = y[:-test_n]
    X_test = X.iloc[-test_n:].reset_index(drop=True)
    y_test = y[-test_n:]

    for idx, row in top.iterrows():
        cand = parse_params(row['params'])
        model_name = row['model']
        outdir = outbase / f"cand_{idx}_{model_name}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Running diagnostics for candidate {idx} {model_name} -> {outdir}")
        summary = run_diagnostics_for_candidate(coord, X_train_full, y_train_full, X_test, y_test, cand, model_name, outdir, cv=3)
        (outdir / 'params.json').write_text(json.dumps(cand, indent=2))


if __name__ == '__main__':
    main()
