import json
from pathlib import Path
import tempfile

from scripts.pdf_to_ground_truth import attach_spans_to_gt


def test_attach_spans_to_gt_top_level(monkeypatch, tmp_path):
    # create minimal GT file with top-level fields but no spans
    gt = {
        'filename': 'foo.pdf',
        'source': 'data/pdf_analysis/foo.pdf',
        'credit_score': 700,
        'monthly_payments': 1234,
        'credit_freeze': 1,
        'fraud_alert': 0,
        'deceased': 0,
        'inquiries_last_6_months': 2,
        'credit_factors': [{'factor': 'Test Factor'}]
    }
    gt_path = tmp_path / 'user_test_ground_truth_unvalidated.json'
    gt_path.write_text(json.dumps(gt), encoding='utf-8')

    # fake map_file to return empty rows for factors (we're testing top-level mapping)
    def fake_map_file(path):
        mapped = tmp_path / 'user_test.mapped.json'
        mapped.write_text(json.dumps({'mapped': []}), encoding='utf-8')
        return mapped, []

    monkeypatch.setattr('scripts.pdf_to_ground_truth.map_file', fake_map_file)

    # monkeypatch extract_pdf_all_fields to return lines with spans for our top-level values
    def fake_extract(pdf_path, include_spans=False):
        lines = [
            {'page': 0, 'spans': [{'text': 'credit score', 'hex': None}], 'bbox': [1,2,3,4]},
            {'page': 0, 'spans': [{'text': '700', 'hex': '#000000'}], 'bbox': [2,3,4,5]},
            {'page': 0, 'spans': [{'text': '$1234/mo', 'hex': '#111111'}], 'bbox': [3,4,5,6]},
            {'page': 0, 'spans': [{'text': 'Credit Freeze', 'hex': None}], 'bbox': [4,5,6,7]},
            {'page': 0, 'spans': [{'text': 'Yes', 'hex': '#ff0000'}], 'bbox': [5,6,7,8]},
            {'page': 0, 'spans': [{'text': 'Inquires (last 6 months)', 'hex': None}], 'bbox': [6,7,8,9]},
            {'page': 0, 'spans': [{'text': '2', 'hex': None}], 'bbox': [7,8,9,10]},
        ]
        return {'all_lines_obj': lines}

    monkeypatch.setattr('scripts.pdf_to_ground_truth.extract_pdf_all_fields', fake_extract)

    enriched = attach_spans_to_gt(gt_path, Path('data/pdf_analysis/foo.pdf'))
    enriched_data = json.loads(enriched.read_text(encoding='utf-8'))

    assert 'credit_score_bbox' in enriched_data and enriched_data['credit_score_bbox'] == [2,3,4,5]
    assert 'monthly_payments_bbox' in enriched_data and enriched_data['monthly_payments_bbox'] == [3,4,5,6]
    assert 'credit_freeze_bbox' in enriched_data and enriched_data['credit_freeze_bbox'] == [5,6,7,8]
    assert 'inquiries_last_6_months_bbox' in enriched_data and enriched_data['inquiries_last_6_months_bbox'] == [7,8,9,10]
