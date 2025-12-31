#!/usr/bin/env python3
"""Compare KS (train vs test) across different split strategies.

Writes:
 - ks_summary.csv with per-strategy mean/median/max KS
 - ks_per_feature.csv with per-feature KS per strategy

Usage: python scripts/ks_split_comparison.py --data-csv data/prefi_weaviate_clean-1_flattened.csv --outdir models/ks_split_comparison
"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import sanitize_column_name
from src.ingest import _load_golden_schema, _parse_schema, preprocess_dataframe
from src.utils import load_column_headers


def ks_numeric_pairs(Xa: pd.DataFrame, Xb: pd.DataFrame, numeric_cols):
    out = {}
    for c in numeric_cols:
        try:
            s = ks_2samp(Xa[c].dropna(), Xb[c].dropna()).statistic
        except Exception:
            s = np.nan
        out[c] = float(s) if not np.isnan(s) else None
    return out


def contiguous_tail_split(df, test_fraction):
    n = len(df)
    test_n = max(100, int(test_fraction * n))
    train_df = df.iloc[:-test_n].reset_index(drop=True)
    test_df = df.iloc[-test_n:].reset_index(drop=True)
    return train_df, test_df


def time_based_split(df, time_col, test_fraction):
    if time_col is None or time_col not in df.columns:
        return None, None
    d = df.copy()
    d = d.sort_values(by=time_col)
    return contiguous_tail_split(d, test_fraction)


def binned_high_ks_split(df, target_col, numeric_cols, high_ks_cols, test_fraction, n_bins=5):
    d = df.copy()
    # create binned columns for each high KS numeric col
    for c in high_ks_cols:
        try:
            d[c + '__bin'] = pd.qcut(d[c].rank(method='first'), q=n_bins, labels=False, duplicates='drop').astype(str)
        except Exception:
            # fallback: equal-width
            d[c + '__bin'] = pd.cut(d[c], bins=n_bins, labels=False).astype(str)
    # build stratify key from target and binned features
    strat_cols = [target_col] + [c + '__bin' for c in high_ks_cols]
    d['strat_key'] = d[strat_cols].astype(str).agg('__'.join, axis=1)

    # ensure no tiny strata: duplicate rows in tiny strata
    counts = d['strat_key'].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        d = pd.concat([d, d[d['strat_key'].isin(rare)]])
        d = d.reset_index(drop=True)

    # If there are more strata than available test slots, group rare strata into a single bucket
    test_n = max(100, int(test_fraction * len(d)))
    counts = d['strat_key'].value_counts()
    # progressively merge low-frequency strata until we have <= test_n unique strata
    while counts.shape[0] > test_n:
        # find the smallest non-zero count
        thresh = counts.min()
        # group all strata with count <= thresh into 'RARE'
        rare_keys = counts[counts <= thresh].index.tolist()
        if len(rare_keys) == 0:
            break
        d.loc[d['strat_key'].isin(rare_keys), 'strat_key'] = 'RARE'
        counts = d['strat_key'].value_counts()
        # If everything collapsed to one bucket, break to avoid infinite loop
        if counts.shape[0] <= 1:
            break

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(d, test_size=test_fraction, stratify=d['strat_key'], random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=Path, default=Path('data/prefi_weaviate_clean-1_flattened.csv'))
    parser.add_argument('--outdir', type=Path, default=Path('models/ks_split_comparison'))
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--ks-threshold', type=float, default=0.20)
    parser.add_argument('--n-bins', type=int, default=5)
    parser.add_argument('--time-col', type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # load + preprocess like gap_diagnostics
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

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    # 1) no_change (contiguous tail)
    tr0, te0 = contiguous_tail_split(full, args.test_fraction)
    ks0 = ks_numeric_pairs(tr0[numeric_cols], te0[numeric_cols], numeric_cols)

    # pick high KS columns from this baseline
    ks_series0 = pd.Series(ks0).dropna()
    high_ks_cols = ks_series0[ks_series0 >= args.ks_threshold].sort_values(ascending=False).index.tolist()
    if len(high_ks_cols) == 0:
        # fallback: pick top 5
        high_ks_cols = ks_series0.sort_values(ascending=False).head(5).index.tolist()

    # 2) time-based
    # Auto-detect a time column if not provided
    time_col = args.time_col
    if not time_col:
        candidates = [c for c in full.columns if c.endswith('_created_at') or 'date' in c.lower() or 'time' in c.lower()]
        time_col = candidates[0] if candidates else None
    if time_col and time_col in full.columns:
        tr_time, te_time = time_based_split(full, time_col, args.test_fraction)
        tr_time = tr_time.reset_index(drop=True)
        te_time = te_time.reset_index(drop=True)
        ks_time = ks_numeric_pairs(tr_time[numeric_cols], te_time[numeric_cols], numeric_cols)
    else:
        # fallback to contiguous tail but mark as fallback
        tr_time, te_time = tr0, te0
        ks_time = ks0

    # 3) binned_high_ks
    tr_bin, te_bin = binned_high_ks_split(full, status_target, numeric_cols, high_ks_cols, args.test_fraction, n_bins=args.n_bins)
    ks_bin = ks_numeric_pairs(tr_bin[numeric_cols], te_bin[numeric_cols], numeric_cols)

    # aggregate
    def summarize(ksdict):
        s = pd.Series(ksdict).dropna()
        return {'mean_ks': float(s.mean()), 'median_ks': float(s.median()), 'max_ks': float(s.max()), 'n_numeric': int(len(s))}

    summary = {
        'no_change': summarize(ks0),
        'time_based': summarize(ks_time),
        'binned_high_ks': summarize(ks_bin),
        'high_ks_cols_used_for_binning': high_ks_cols
    }

    # write per-feature table
    ks_df = pd.DataFrame({'no_change': ks0, 'time_based': ks_time, 'binned_high_ks': ks_bin})
    ks_df.index.name = 'feature'
    ks_df.to_csv(outdir / 'ks_per_feature.csv')

    # write summary (flattened table)
    summary_rows = []
    for name, val in summary.items():
        if name == 'high_ks_cols_used_for_binning':
            continue
        r = {'strategy': name}
        r.update(val)
        summary_rows.append(r)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(outdir / 'ks_summary.csv', index=False)

    (outdir / 'ks_summary.json').write_text(json.dumps(summary, indent=2))

    print('Wrote results to', outdir)


if __name__ == '__main__':
    main()
