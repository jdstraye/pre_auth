#!/usr/bin/env python3
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Intensive model search runner')
    parser.add_argument('--output-dir', type=str, default='models/intensive_search')
    parser.add_argument('--n-top', type=int, default=1)
    parser.add_argument('--n-samples-per-model', type=int, default=50)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--flush-every', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-csv', type=str, default='data/prefi_weaviate_clean-1_flattened.csv', help='Path to input CSV data file')
    parser.add_argument('--column-headers', type=str, default='src/column_headers.json', help='Path to column headers JSON')
    parser.add_argument('--exhaustive', action='store_true', help='Use exhaustive grid search instead of random sampling')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of parameter combinations to try (optional)')
    # Remove --no-progress from CLI, always use progress bar
    args = parser.parse_args()

    # Always add project root and src to sys.path
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'src'))

    from src import run_exhaustive_search

    # Always use progress bar
    args.no_progress = False

    # Add advanced boosting techniques to param grid
    from src.eval_algos import param_distributions
    # LightGBM: try 'dart' and 'goss' boosting types for better regularization and handling of overfitting
    if 'LGBMClassifier' in param_distributions:
        param_distributions['LGBMClassifier']['boosting_type'] = ['gbdt', 'dart', 'goss']
    # XGBoost: try 'dart' booster for dropout regularization in addition to standard 'gbtree'
    if 'XGBClassifier' in param_distributions:
        param_distributions['XGBClassifier']['booster'] = ['gbtree', 'dart']

    run_exhaustive_search.run_search(args)
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
    exhaustive = importlib.import_module('src.run_exhaustive_search')
    exhaustive.run_search(args)


if __name__ == '__main__':
    main()
