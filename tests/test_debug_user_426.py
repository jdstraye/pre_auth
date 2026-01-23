def test_debug_user_426_expected_green_phrases():
    from src.scripts.pdf_color_extraction import color_first_search_for_phrase
    from src.utils import map_color_to_cat
    import pytest
    pytest.importorskip('fitz')
    from src.pymupdf_compat import fitz

    pdf = 'data/pdf_analysis/user_426_credit_summary_2025-09-01_095930.pdf'
    phrases_expected_green = [
        'Ok Open Rev Depth',
        'Past Due Not Late',
        '4 RE Lates in 4-6 mo',
    ]

    phrases_expected_black = [
        'No Open Mortgage',
        'No Rev Acct Open 10K 2yr',
    ]

    phrases_expected_red = [
        '1 Rev Late in 0-3 mo',
        '2 RE Lates in 6-12 mo',
        'No Closed Rev Depth',
        'Avg Age Open',
        'No 7.5k+ Lines',
    ]

    doc = fitz.open(pdf)

    # Check greens
    for phrase in phrases_expected_green:
        res = color_first_search_for_phrase(doc, phrase, page_limit=3)
        assert res is not None, f"color_first_search_for_phrase returned None for '{phrase}'"
        pidx, text, hexv, rgb, bbox, method, _ = res
        mapped = map_color_to_cat(rgb) if rgb else None
        print(phrase, '=>', {'text': text, 'hex': hexv, 'rgb': rgb, 'mapped_cat': mapped})
        assert mapped == 'green', f"Expected green for '{phrase}', got {mapped} (text='{text}')"

    # Check blacks
    for phrase in phrases_expected_black:
        res = color_first_search_for_phrase(doc, phrase, page_limit=3)
        assert res is not None, f"color_first_search_for_phrase returned None for '{phrase}'"
        pidx, text, hexv, rgb, bbox, method, _ = res
        mapped = map_color_to_cat(rgb) if rgb else None
        print(phrase, '=>', {'text': text, 'hex': hexv, 'rgb': rgb, 'mapped_cat': mapped})
        assert mapped == 'black', f"Expected black for '{phrase}', got {mapped} (text='{text}')"

    # Check reds
    for phrase in phrases_expected_red:
        res = color_first_search_for_phrase(doc, phrase, page_limit=3)
        assert res is not None, f"color_first_search_for_phrase returned None for '{phrase}'"
        pidx, text, hexv, rgb, bbox, method, _ = res
        mapped = map_color_to_cat(rgb) if rgb else None
        print(phrase, '=>', {'text': text, 'hex': hexv, 'rgb': rgb, 'mapped_cat': mapped})
        assert mapped == 'red', f"Expected red for '{phrase}', got {mapped} (text='{text}')"
