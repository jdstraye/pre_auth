import json
from pathlib import Path

from scripts.promote_validated_gt import build_validated_payload


def test_build_payload_requires_spans_and_bbox():
    # mapped-like minimal structure with one matched row
    mapped = {
        'source_gt': 'data/extracted/user_1314_credit_summary_2025-09-01_092724_ground_truth.json',
        'mapped': [
            {
                'factor': 'Drop Bad Auth User (BEST BUY/CBNA, and CO',
                'color': '#212529',
                'canonical_key': 'drop_bad_auth_user_best_buy_cbna_and_co--abcd',
                'page': 0,
                'bbox': [10, 20, 300, 40],
                'spans': [{'text': 'Drop Bad Auth User (BEST BUY/CBNA, and CO', 'hex': '#212529', 'font_size': 10, 'is_bold': False}],
                'match_type': 'exact',
                'match_score': 1.0
            }
        ]
    }
    payload = build_validated_payload(mapped, {0})
    assert 'credit_factors' in payload
    assert payload['credit_factors'][0]['bbox'] == [10, 20, 300, 40]
    assert payload['credit_factors'][0]['spans'][0]['text'].startswith('Drop Bad Auth')
