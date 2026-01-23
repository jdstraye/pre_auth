"""Validate ground-truth JSON(s) for required fields used by the refactor.

Checks performed (strict):
- `credit_factors` exists and is a list
- each factor has `factor` (string)
- if `page`/`bbox`/`spans` present, they must be well-formed; otherwise flagged as missing
- canonical_key presence is recommended (warning)

Exit code: 0 if all files pass required checks; non-zero otherwise.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

# Allow running as `python scripts/validate_ground_truth.py` without setting PYTHONPATH explicitly
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FACTOR_KEYS = ['factor']
RECOMMENDED_FACTOR_KEYS = ['canonical_key']


def validate_file(path: Path) -> List[str]:
    errors: List[str] = []
    j = json.loads(path.read_text(encoding='utf-8'))
    factors = j.get('credit_factors')
    if not isinstance(factors, list):
        errors.append('missing_or_invalid:credit_factors')
        return errors
    for i, f in enumerate(factors):
        if not isinstance(f, dict):
            errors.append(f'factor[{i}]:not_object')
            continue
        for k in REQUIRED_FACTOR_KEYS:
            if k not in f or not isinstance(f.get(k), str) or not f.get(k).strip():
                errors.append(f'factor[{i}]:missing_or_empty:{k}')
        # page/bbox/spans are required for full-refactor; flag if missing
        if 'page' not in f:
            errors.append(f'factor[{i}]:missing:page')
        else:
            if not isinstance(f.get('page'), int):
                errors.append(f'factor[{i}]:invalid:page')
        if 'bbox' not in f:
            errors.append(f'factor[{i}]:missing:bbox')
        else:
            bb = f.get('bbox')
            if not (isinstance(bb, list) and len(bb) == 4):
                errors.append(f'factor[{i}]:invalid:bbox')
        if 'spans' not in f:
            errors.append(f'factor[{i}]:missing:spans')
        else:
            if not isinstance(f.get('spans'), list):
                errors.append(f'factor[{i}]:invalid:spans')
        if 'canonical_key' not in f:
            errors.append(f'factor[{i}]:missing:canonical_key (recommended)')
    return errors


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args(argv)
    any_err = False
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"MISSING: {p}")
            any_err = True
            continue
        errs = validate_file(path)
        if errs:
            any_err = True
            print(f"{p}: FAIL")
            for e in errs:
                print("  -", e)
        else:
            print(f"{p}: OK")
    return 1 if any_err else 0


if __name__ == '__main__':
    raise SystemExit(main())
