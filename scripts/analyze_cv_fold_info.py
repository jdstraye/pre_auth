#!/usr/bin/env python3
"""Analyze CV fold-level metadata produced by `run_exhaustive_search.py`.

Reads a search_results CSV (or any DataFrame with a `cv_fold_info` column containing
JSON) and prints a short summary per candidate showing SMOTE per-fold behavior and
selected feature stability.
"""
from pathlib import Path
import json
import argparse
import pandas as pd
from collections import Counter, defaultdict


def summarize_cv_fold_info(df: pd.DataFrame, top_n: int = 10):
    """Return a list of summaries for top `top_n` candidates in df (assumes df sorted by mean_f1)."""
    summaries = []
    if df.empty:
        return summaries
    for i, row in enumerate(df.head(top_n).to_dict(orient='records')):
        ts = row.get('timestamp')
        model = row.get('model')
        mean_f1 = row.get('mean_f1')
        raw = row.get('cv_fold_info', '[]')
        try:
            folds = json.loads(raw)
        except Exception:
            folds = []
        total_folds = len(folds)
        smote_enabled_count = sum(1 for f in folds if f.get('smote_enabled'))
        min_class_le1 = sum(1 for f in folds if int(f.get('min_class_count', 0)) <= 1)
        # selected features stability
        feat_counts = Counter()
        for f in folds:
            for feat in f.get('selected_features', []) or []:
                feat_counts[str(feat)] += 1
        top_feats = feat_counts.most_common(10)
        summaries.append({
            'timestamp': ts,
            'model': model,
            'mean_f1': mean_f1,
            'total_folds': total_folds,
            'smote_enabled_count': smote_enabled_count,
            'folds_min_class_le1': min_class_le1,
            'top_selected_features': top_feats
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description='Summarize CV fold-level info from search results')
    parser.add_argument('--results-csv', type=Path, required=True, help='Path to search_results.csv')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top candidates to summarize')
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    # sort by mean_f1 desc if present
    if 'mean_f1' in df.columns:
        df = df.sort_values(by='mean_f1', ascending=False)

    summaries = summarize_cv_fold_info(df, top_n=args.top_n)
    for s in summaries:
        try:
            meanf = float(s['mean_f1']) if s['mean_f1'] is not None else None
            meanf_str = f"{meanf:.4f}" if meanf is not None else 'None'
        except Exception:
            meanf_str = str(s['mean_f1'])
        print(f"[{s['timestamp']}] {s['model']} mean_f1={meanf_str} folds={s['total_folds']} smote_on_folds={s['smote_enabled_count']} folds_min_class<=1={s['folds_min_class_le1']}")
        if s['top_selected_features']:
            print("  top selected features (count):", ", ".join(f"{k}:{v}" for k, v in s['top_selected_features']))
        print()


if __name__ == '__main__':
    main()
