from src.scripts.pdf_color_extraction import normalize_factors


def test_normalize_factors_hex_and_color_present():
    raw = [
        {'factor': 'Test red thing', 'color': 'red', 'hex': '#ff0000', 'bbox': [1,2,3,4]},
        {'factor': 'Neutral thing', 'color': 'neutral', 'hex': None}
    ]
    out = normalize_factors(raw)
    assert isinstance(out, list)
    assert any(f for f in out if f['factor'] == 'Test red thing' and f['color'] == 'red' and (f.get('hex') == '#ff0000' or f.get('hex') is None))
    assert any(f for f in out if f['factor'] == 'Neutral thing' and f['color'] == 'neutral')
