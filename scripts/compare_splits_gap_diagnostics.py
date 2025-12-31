#!/usr/bin/env python3
"""Run gap diagnostics for two split strategies (baseline contiguous tail and bin_high_ks)
and compare CV->test gaps for top candidates.

Outputs:
 - models/gap_diagnostics_compare/compare_table.csv
 - per-candidate outputs under models/gap_diagnostics_compare/cand_<idx>_<model>/{baseline,binned}/summary.json
"""
from pathlib import Path
import json
import argparse
import sys
import pandas as pd
import numpy as np
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gap_diagnostics import run_diagnostics_for_candidate, parse_params
from src.pipeline_coordinator import create_default_coordinator
from src.components.smote_sampler import MaybeSMOTESampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import ks_2samp


def contiguous_tail_split(full_df, test_fraction):
    n = len(full_df)
    test_n = max(100, int(test_fraction * n))
    train_df = full_df.iloc[:-test_n].reset_index(drop=True)
    test_df = full_df.iloc[-test_n:].reset_index(drop=True)
    return train_df, test_df


def binned_high_ks_split(full_df, status_target, test_fraction, ks_threshold=0.2, n_bins=5):
    tr0, te0 = contiguous_tail_split(full_df, test_fraction)
    num_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c])]
    ks_vals = {}
    for c in num_cols:
        try:
            ks_vals[c] = float(ks_2samp(tr0[c].dropna(), te0[c].dropna()).statistic)
        except Exception:
            ks_vals[c] = 0.0
    high_ks = [k for k, v in sorted(ks_vals.items(), key=lambda it: -it[1]) if v >= ks_threshold][:10]
    if not high_ks:
        high_ks = [k for k, v in sorted(ks_vals.items(), key=lambda it: -it[1])][:5]

    d = full_df.copy()
    for c in high_ks:
        bname = f"{c}__bin"
        try:
            d[bname] = pd.qcut(d[c].rank(method='first'), q=n_bins, labels=False, duplicates='drop').astype(str)
        except Exception:
            d[bname] = pd.cut(d[c], bins=n_bins, labels=False).astype(str)
    # stratify key
    strat_cols = [status_target] + [f"{c}__bin" for c in high_ks]
    d['strat_key'] = d[ strat_cols ].astype(str).agg('__'.join, axis=1)

    from sklearn.model_selection import train_test_split
    # collapse rare strata if needed
    counts = d['strat_key'].value_counts()
    test_n = max(100, int(test_fraction * len(d)))
    while counts.shape[0] > test_n:
        thresh = counts.min()
        rare_keys = counts[counts <= thresh].index.tolist()
        if not rare_keys:
            break
        d.loc[d['strat_key'].isin(rare_keys), 'strat_key'] = 'RARE'
        counts = d['strat_key'].value_counts()
        if counts.shape[0] <= 1:
            break

    tr, te = train_test_split(d, test_size=test_fraction, stratify=d['strat_key'], random_state=42)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def fit_and_test_full(coord, cand, Xtr, ytr, Xte, yte, model_name):
    model_ctor = RandomForestClassifier
    base_est = clone(model_ctor())
    p = coord.create_pipeline(base_estimator=clone(base_est), smote_config={"enabled": bool(cand.get('smote__enabled', True)), "categorical_feature_names": cand.get('smote__categorical_feature_names', []), "k_neighbors": int(cand.get('smote__k_neighbors', 5)), "allow_fallback": True}, feature_selection_config={'max_features': cand.get('feature_selecting_classifier__max_features', None), 'threshold': cand.get('feature_selecting_classifier__threshold', None)})
    try:
        p.set_params(**cand)
    except Exception:
        pass
    p.fit(Xtr, ytr)
    ypred = p.predict(Xte)
    from sklearn.metrics import f1_score
    return float(f1_score(yte, ypred, average='macro'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--results-dir', type=Path, default=Path('models/gap_diagnostics_compare'))
    parser.add_argument('--data-csv', type=Path, default=Path('data/prefi_weaviate_clean-1_flattened.csv'))
    parser.add_argument('--ks-threshold', type=float, default=0.20)
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

    full = pd.concat([X, pd.Series(y, name=status_target)], axis=1)

    rows = []

    for idx, row in top.iterrows():
        cand = parse_params(row['params'])
        model_name = row['model']
        outdir = outbase / f"cand_{idx}_{model_name}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"Running compare diagnostics for candidate {idx} {model_name}")

        # baseline split
        trb, teb = contiguous_tail_split(full, 0.2)
        Xtrb = trb[feature_cols].reset_index(drop=True)
        ytrb = trb[status_target].astype(int).to_numpy()
        Xteb = teb[feature_cols].reset_index(drop=True)
        yteb = teb[status_target].astype(int).to_numpy()

        # binned split
        trbin, tebin = binned_high_ks_split(full, status_target, 0.2, ks_threshold=args.ks_threshold)
        Xtrbin = trbin[feature_cols].reset_index(drop=True)
        ytrbin = trbin[status_target].astype(int).to_numpy()
        Xtebin = tebin[feature_cols].reset_index(drop=True)
        ytebin = tebin[status_target].astype(int).to_numpy()

        # run diagnostics for each
        base_dir = outdir / 'baseline'
        base_dir.mkdir(exist_ok=True)
        bsum = run_diagnostics_for_candidate(coord, Xtrb, ytrb, Xteb, yteb, cand, model_name, base_dir, cv=3)
        (base_dir / 'params.json').write_text(json.dumps(cand, indent=2))

        b_test_f1 = fit_and_test_full(coord, cand, Xtrb, ytrb, Xteb, yteb, model_name)
        b_cv_mean = np.mean([r['fold_f1'] for r in bsum['fold_records']])
        b_gap = float(b_cv_mean) - float(b_test_f1)

        bin_dir = outdir / 'binned'
        bin_dir.mkdir(exist_ok=True)
        bnsum = run_diagnostics_for_candidate(coord, Xtrbin, ytrbin, Xtebin, ytebin, cand, model_name, bin_dir, cv=3)
        (bin_dir / 'params.json').write_text(json.dumps(cand, indent=2))

        bin_test_f1 = fit_and_test_full(coord, cand, Xtrbin, ytrbin, Xtebin, ytebin, model_name)
        bin_cv_mean = np.mean([r['fold_f1'] for r in bnsum['fold_records']])
        bin_gap = float(bin_cv_mean) - float(bin_test_f1)

        rows.append({'candidate_idx': int(idx), 'model': model_name,
                     'baseline_cv_mean': float(b_cv_mean), 'baseline_test_f1': float(b_test_f1), 'baseline_gap': float(b_gap),
                     'binned_cv_mean': float(bin_cv_mean), 'binned_test_f1': float(bin_test_f1), 'binned_gap': float(bin_gap)})

    pd.DataFrame(rows).to_csv(outbase / 'compare_table.csv', index=False)
    print('Wrote comparison to', outbase / 'compare_table.csv')


if __name__ == '__main__':
    main()
