from scripts.pdf_to_ground_truth import build_text_only_gt


def test_build_text_only_gt_derives_color_from_hex_and_spans():
    rec = {
        'filename': 'foo.pdf',
        'source': 'data/pdf_analysis/foo.pdf',
        'credit_factors': [
            {
                'factor': 'Test Red',
                'hex': '#ff0000',
                'spans': [{'text': 'Test Red', 'hex': '#ff0000', 'rgb': [255,0,0], 'bbox': [1,2,3,4]}]
            },
            {
                'factor': 'Test Neutral',
                'color': 'neutral'
            },
            {
                'factor': 'Test NamedHex',
                'hex': 'red'
            }
        ]
    }
    out = build_text_only_gt(rec, include_spans=True)
    cfs = {c['factor']: c for c in out['credit_factors']}
    assert cfs['Test Red']['color'] == 'red'
    assert cfs['Test Neutral']['color'] == 'neutral'
    # 'hex' was a named color string; we expect that to be normalized into 'color'
    assert cfs['Test NamedHex']['color'] == 'red'
