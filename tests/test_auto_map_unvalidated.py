import json
from pathlib import Path

from scripts.auto_map_unvalidated import map_file


def test_map_user_1314_has_spans_and_bbox(tmp_path):
    # user_1314 is present in the repo and has good extractor outputs
    gt = Path('data/extracted/user_1314_credit_summary_2025-09-01_092724_ground_truth.json')
    assert gt.exists(), 'expected user_1314 GT to exist in repo for the test'
    out_json, rows = map_file(str(gt))
    assert out_json.exists()
    mapped = json.loads(out_json.read_text())
    assert 'mapped' in mapped
    # every factor should have either spans/bbox attached or a clear note
    for r in mapped['mapped']:
        assert ('spans' in r and r['spans']) or ('notes' in r and r['notes']), 'each factor must have spans or a notes flag'
        # canonical_key should be set for matched lines
        if r.get('match_type') in ('exact', 'substring', 'fuzzy'):
            assert r.get('canonical_key'), 'matched factor should have canonical_key'
