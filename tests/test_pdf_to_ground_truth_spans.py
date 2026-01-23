import json
from pathlib import Path
import tempfile

from scripts.pdf_to_ground_truth import build_text_only_gt, attach_spans_to_gt


def test_build_text_only_gt_preserves_spans():
    rec = {
        'filename': 'foo.pdf',
        'source': 'data/pdf_analysis/foo.pdf',
        'credit_factors': [
            {
                'factor': 'Test Factor',
                'hex': 'red',
                'bbox': [1, 2, 3, 4],
                'page': 0,
                'spans': [{'text': 'Test Factor', 'hex': 'red', 'bbox': [1,2,3,4]}]
            }
        ]
    }
    out = build_text_only_gt(rec)
    assert 'credit_factors' in out
    cf = out['credit_factors'][0]
    assert cf['factor'] == 'Test Factor'
    # ensure bbox/page/spans copied through
    assert cf['bbox'] == [1, 2, 3, 4]
    assert cf['page'] == 0
    assert isinstance(cf['spans'], list)


def test_attach_spans_to_gt_merges_map(monkeypatch, tmp_path):
    # create a minimal gt file
    gt = {
        'filename': 'foo.pdf',
        'source': 'data/pdf_analysis/foo.pdf',
        'credit_factors': [{'factor': 'Test Factor'}]
    }
    gt_path = tmp_path / 'user_test_ground_truth_unvalidated.json'
    gt_path.write_text(json.dumps(gt), encoding='utf-8')

    # fake map_file to return rows with page/bbox/spans/canonical_key
    fake_rows = [
        {'factor': 'Test Factor', 'page': 2, 'bbox': [10,11,12,13], 'spans': [{'text':'Test Factor','hex':'red'}], 'canonical_key': 'test_factor--abc123'}
    ]

    def fake_map_file(path):
        # create fake mapped json file path
        mapped = tmp_path / 'user_test.mapped.json'
        mapped.write_text(json.dumps({'mapped': fake_rows}), encoding='utf-8')
        return mapped, fake_rows

    monkeypatch.setattr('scripts.pdf_to_ground_truth.map_file', fake_map_file)

    enriched = attach_spans_to_gt(gt_path, Path('data/pdf_analysis/foo.pdf'))
    enriched_data = json.loads(enriched.read_text(encoding='utf-8'))
    assert 'credit_factors' in enriched_data
    cf = enriched_data['credit_factors'][0]
    assert cf['page'] == 2
    assert cf['bbox'] == [10,11,12,13]
    assert cf['spans'][0]['text'] == 'Test Factor'
    assert cf['canonical_key'] == 'test_factor--abc123'
