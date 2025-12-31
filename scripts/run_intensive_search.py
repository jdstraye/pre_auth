#!/usr/bin/env python3
"""
Run an intensive hyperparameter search wrapper around run_exhaustive_search.

This script is a thin wrapper that sets denser defaults (more samples/exhaustive) and
targets the `MLPipelineCoordinator` pipeline flow: ingest -> allocate -> pipeline_coordinator.

Outputs are written to `models/intensive_search` by default and include per-candidate
metrics (accuracy, precision, recall, f1) and feature importance information.
"""
from pathlib import Path
import argparse
exhaustive = None  # imported lazily in main to avoid heavy import-time deps
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Intensive hyperparameter search runner (wrapper)")
    parser.add_argument("--data-csv", type=Path, default=ROOT / "data" / "prefi_weaviate_clean-1_flattened.csv")
    parser.add_argument("--start-json", type=Path, default=None, help='Start from raw JSON (will run ingest + preprocess)')
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--column-headers", type=Path, default=ROOT / "src" / "column_headers.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "intensive_search")
    parser.add_argument("--n-top", type=int, default=10)
    parser.add_argument("--n-samples-per-model", type=int, default=200, dest="n_samples_per_model",
                        help='Number of parameter samples to draw per model (higher -> more intensive)')
    parser.add_argument("--random-search-mult", type=float, default=0.5, dest='random_search_mult',
                        help='Multiplier for default random search iterations per model to scale sampling density')
    parser.add_argument("--save-format", type=str, default='csv', choices=['csv', 'sqlite'], dest='save_format')
    parser.add_argument("--db-path", type=str, default='intensive_search_results.db', dest='db_path')
    parser.add_argument("--flush-every", type=int, default=5, dest='flush_every')
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of combos per model; 0 means no limit")
    parser.add_argument("--exhaustive", action='store_true', help="Enumerate all parameter combinations (cartesian product)")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no-progress", action='store_true', default=False, help='Disable inline progress bar and ETA')
    parser.add_argument("--target-f1", type=float, default=None, help='Optional early-stop target F1 score (macro)')
    parser.add_argument("--dry-run", action='store_true', help='Perform a quick dry-run without executing the full search')

    args = parser.parse_args()

    # Re-use the exhaustive search runner with denser defaults
    # set output dir
    args.output_dir = args.output_dir
    # Propagate the target F1 through args (script uses it in coordinator.search_models if supported)
    setattr(args, 'target_f1', args.target_f1)

    # Dry-run path used primarily for testing without heavy runtime deps
    if getattr(args, 'dry_run', False):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # write an empty search_results.csv to emulate output
        import pandas as _pd
        _pd.DataFrame([]).to_csv(args.output_dir / 'search_results.csv', index=False)
        return

    # Import the exhaustive search implementation lazily to avoid heavy import-time deps
    import importlib
    exhaustive = importlib.import_module('scripts.run_exhaustive_search')
    exhaustive.run_search(args)


if __name__ == '__main__':
    main()
