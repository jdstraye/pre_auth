#!/usr/bin/env python3
def main():
    import argparse
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'src'))

    parser = argparse.ArgumentParser(description='Intensive model search runner')
    parser.add_argument('--data-csv', type=Path, default=ROOT / 'data' / 'prefi_weaviate_clean-1_flattened.csv', help='Path to input CSV data file')
    parser.add_argument('--column-headers', type=Path, default=ROOT / 'src' / 'column_headers.json', help='Path to column headers JSON')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models' / 'intensive_search')
    parser.add_argument('--n-top', type=int, default=5)
    parser.add_argument('--n-samples-per-model', type=int, default=50)
    parser.add_argument('--flush-every', type=int, default=10)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--exhaustive', action='store_true', help='Use exhaustive grid search instead of random sampling')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Always use progress bar
    args.no_progress = False

    # Add advanced boosting techniques to param grid
    from src.eval_algos import param_distributions
    if 'LGBMClassifier' in param_distributions:
        param_distributions['LGBMClassifier']['boosting_type'] = ['gbdt', 'dart', 'goss']
    if 'XGBClassifier' in param_distributions:
        param_distributions['XGBClassifier']['booster'] = ['gbtree', 'dart']

    from src import run_exhaustive_search
    run_exhaustive_search.run_search(args)


if __name__ == '__main__':
    main()
