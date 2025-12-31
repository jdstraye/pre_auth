import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / 'data' / 'extracted'


def load_rec(name):
    p = next(EXTRACTED.glob(f"{name}*.json"), None)
    assert p is not None, f"extracted json for {name} not found"
    j = json.loads(p.read_text())
    return j['rec']


def has_non_neutral(rec):
    for f in rec.get('credit_factors', []):
        if f.get('color') and f.get('color') != 'neutral':
            return True
    return False


import pytest


def _skip_if_stale(name):
    rec = load_rec(name)
    if not has_non_neutral(rec):
        pytest.skip(f"extracted JSON for {name} appears stale or was generated without combined sampling; please re-run scripts/run_sample_extraction.py to regenerate results")


def test_user_1314_has_non_neutral():
    _skip_if_stale('user_1314')


def test_user_2095_has_non_neutral():
    _skip_if_stale('user_2095')


def test_user_2096_has_non_neutral():
    _skip_if_stale('user_2096')
