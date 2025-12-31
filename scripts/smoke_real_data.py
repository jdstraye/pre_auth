#!/usr/bin/env python3
"""Smoke-run helper for real data: dry-run and limited-intensity run."""
from pathlib import Path
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-json', type=Path, default=Path('data/prefi_weaviate_clean-2.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--n-samples-per-model', type=int, default=10)
    parser.add_argument('--n-top', type=int, default=5)
    args = parser.parse_args()

    ris = importlib.import_module('scripts.run_intensive_search')
    # Build argv for the wrapper
    argv = ['run_intensive_search.py', '--start-json', str(args.start_json), '--n-samples-per-model', str(args.n_samples_per_model), '--n-top', str(args.n_top)]
    if args.dry_run:
        argv.append('--dry-run')
    import sys
    sys.argv = argv
    ris.main()

if __name__ == '__main__':
    main()
