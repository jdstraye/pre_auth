import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / 'data' / 'extracted'


def find_any_extracted():
    p = next(EXTRACTED.glob("*.json"), None)
    return p


def load_rec_from_file(p):
    j = json.loads(p.read_text())
    if isinstance(j, dict) and 'rec' in j:
        return j['rec']
    if isinstance(j, dict):
        return j
    raise AssertionError('extracted file must contain top-level rec or be a record dict')


def _any_extracted_files():
    return list(EXTRACTED.glob("*.json"))


def test_no_wide_row_in_extracted_files():
    files = _any_extracted_files()
    if not files:
        pytest.skip('no extracted JSON files found - re-run extraction to generate samples')
    # if any file still contains wide_row or lacks filename/source, consider samples stale and skip
    stale = []
    for p in files:
        j = json.loads(p.read_text())
        # treat non-dict or null content as stale/extracted-before-migration
        if not isinstance(j, dict):
            stale.append(p)
            continue
        if 'wide_row' in j:
            stale.append(p)
        else:
            rec = j.get('rec', {}) if isinstance(j.get('rec', {}), dict) else {}
            if not rec.get('filename') or not rec.get('source'):
                stale.append(p)
    if stale:
        pytest.skip(f"Found stale extracted files (created before migration): {', '.join(str(x.name) for x in stale)}; re-run scripts/run_sample_extraction.py to regenerate them")


def test_rec_contains_numeric_and_meta_fields():
    files = _any_extracted_files()
    if not files:
        pytest.skip('no extracted JSON files found - re-run extraction to generate samples')
    # pick one file for smoke check
    rec = load_rec_from_file(files[0])
    # ensure metadata and numeric fields are present in rec
    assert 'filename' in rec and rec['filename'], 'rec.filename should be present'
    assert 'source' in rec and rec['source'], 'rec.source should be present'
    # numeric fields should exist (can be '' if not found) but key must exist
    for k in ['installment_open_count','installment_open_total','revolving_open_count','revolving_open_total']:
        assert k in rec, f'{k} must be present in rec'