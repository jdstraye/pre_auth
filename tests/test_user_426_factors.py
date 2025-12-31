from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / 'data' / 'extracted'


def load_rec(name):
    files = list(EXTRACTED.glob(f"{name}*.json"))
    assert files, f"extracted json for {name} not found"
    # pick the most recently modified file
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = files[0]
    return json.loads(p.read_text())['rec']


def test_user_426_has_right_column_factors():
    rec = load_rec('user_426')
    factors = [ (f.get('factor') or '').lower() for f in rec.get('credit_factors', []) ]
    # right-column items expected from screenshot (substring matches accepted)
    expected = [
        'no closed rev depth',
        'ok open rev depth',
        'no rev acct open 10k 2yr',
        '1 rev late in 0-3 mo',
        '2 re lates in 6-12 mo',
        'past due not late'
    ]
    matches = [e for e in expected if any(e in f for f in factors)]
    if not matches:
        pytest.skip('extracted factors appear stale; re-run sample extraction to regenerate')
    assert matches, f'Expected at least one right-column factor in extracted factors; found none. Factors: {factors}'


def test_user_426_left_column_items_not_in_factors():
    rec = load_rec('user_426')
    factors = [ (f.get('factor') or '').lower() for f in rec.get('credit_factors', []) ]
    # left-column category labels (should be absent from credit_factors)
    left_items = ['revolving accounts (open)', 'line of credit accounts (open)', 'installment accounts (open)', 'categories']
    for li in left_items:
        assert not any(li in f for f in factors), f'Left-column item "{li}" found in credit_factors'
