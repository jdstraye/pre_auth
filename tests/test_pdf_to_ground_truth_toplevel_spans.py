import json
from scripts.pdf_to_ground_truth import build_text_only_gt


def test_build_text_only_gt_includes_toplevel_spans_when_requested():
    rec = {
        'filename': 'foo.pdf',
        'source': 'data/pdf_analysis/foo.pdf',
        'monthly_payments': 1234,
        'monthly_payments_bbox': [1, 2, 3, 4],
        'monthly_payments_page': 0,
        'monthly_payments_spans': [{'text': '$1,234/mo', 'hex': '#000000'}],
        'credit_freeze': 1,
        'credit_freeze_bbox': [10, 11, 12, 13],
        'credit_freeze_page': 0,
        'credit_freeze_spans': [{'text': 'Yes', 'hex': '#ff0000'}],
    }
    out = build_text_only_gt(rec, include_spans=True)
    assert out['monthly_payments'] == 1234
    assert out['monthly_payments_bbox'] == [1, 2, 3, 4]
    assert out['monthly_payments_page'] == 0
    assert isinstance(out['monthly_payments_spans'], list)
    assert out['credit_freeze'] == 1
    assert out['credit_freeze_bbox'] == [10, 11, 12, 13]
    assert out['credit_freeze_page'] == 0
    assert isinstance(out['credit_freeze_spans'], list)
