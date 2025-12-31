import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / 'data' / 'extracted'
FIX = ROOT / 'tests' / 'fixtures'


def load_rec(name):
    files = list(EXTRACTED.glob(f"{name}*.json"))
    if not files:
        pytest.skip('extracted JSON not found; run extraction first')
    # pick most recently modified file
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = files[0]
    return json.loads(p.read_text())['rec']


def test_user_426_matches_expected():
    rec = load_rec('user_426')
    expected = json.loads((FIX / 'user_426_expected.json').read_text())
    # basic fields
    assert rec.get('address') == expected['address']
    assert int(rec.get('revolving_open_count',0)) == expected['revolving_open_count']
    assert int(rec.get('revolving_open_total',0)) == expected['revolving_open_total']
    # installment may be string if missing - coerce
    assert int(rec.get('installment_open_count') or 0) == expected['installment_open_count']
    assert int(rec.get('installment_open_total') or 0) == expected['installment_open_total']
    assert int(rec.get('public_records') or 0) == expected['public_records']
    # expected factor phrases present (simplified factor format)
    factors = [ (f.get('factor') or '').lower() for f in rec.get('credit_factors', []) ]
    expected_phrases = set(json.loads((FIX / 'user_426_expected.json').read_text())['expected_factors'])
    matches = [e for e in expected_phrases if any(e in f for f in factors)]
    assert matches, f'Expected at least one expected right-column factor in rec.credit_factors; got {factors}'
    # ensure factor dicts only have 'factor' and 'color' keys
    for f in rec.get('credit_factors', []):
        assert set(f.keys()).issubset({'factor','color'}), f'factor dict contains unexpected keys: {f.keys()}'
    # check colors for specific phrases
    mapping = { (f.get('factor') or '').lower(): f.get('color') for f in rec.get('credit_factors', []) }
    assert mapping.get('past due not late') == 'green'
    assert mapping.get('1 rev late in 0-3 mo') == 'red'
    assert mapping.get('2 re lates in 6-12 mo') == 'red'
    assert mapping.get('no closed rev depth') == 'red'
    assert mapping.get('avg age open') == 'red'
    assert mapping.get('no 7.5k+ lines') == 'red'
    assert mapping.get('ok open rev depth') == 'green'
    assert mapping.get('no open mortgage') == 'neutral'
    assert mapping.get('no rev acct open 10k 2yr') == 'neutral'
    # ensure non-factors are not present
    assert not any('past due' == (f.get('factor') or '').lower() for f in rec.get('credit_factors', []))
    assert not any('payment resp' == (f.get('factor') or '').lower() for f in rec.get('credit_factors', []))
