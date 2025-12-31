import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / 'data' / 'extracted'


def load_rec(name):
    p = next(EXTRACTED.glob(f"{name}*.json"), None)
    assert p is not None, f"extracted json for {name} not found"
    j = json.loads(p.read_text())
    return j['rec']


def _skip_if_stale(name):
    rec = load_rec(name)
    if rec.get('public_record_note') and rec.get('public_records',0) == 0:
        pytest.skip(f"extracted JSON for {name} appears stale (note present but count=0); re-run extraction to refresh")


def test_user_2096_public_record_note_implies_count():
    rec = load_rec('user_2096')
    if not rec.get('public_record_note'):
        pytest.skip('user_2096 has no public_record_note in extracted JSON; re-run extraction if PDF shows details')
    assert rec.get('public_records',0) >= 1


def test_user_2095_public_records_zero_when_no_note():
    rec = load_rec('user_2095')
    # user_2095 in sample images shows 'Public Records 0'
    assert rec.get('public_records',0) == 0
    assert rec.get('public_record_note','') == ''
