import fitz
from scripts.poc_extract_credit_factors import combined_sample_color_for_phrase, map_color_to_cat


def test_user_733_ing_last_4_mo_recovered():
    pdf = 'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
    doc = fitz.open(pdf)
    # Ensure the public name is bound to the impl for deterministic behavior in tests
    try:
        import importlib
        import scripts.poc_extract_credit_factors as pc
        importlib.reload(pc)
        if hasattr(pc, 'combined_sample_color_for_phrase_impl'):
            pc.combined_sample_color_for_phrase = pc.combined_sample_color_for_phrase_impl
    except Exception:
        pass
    # call the module-level function after ensuring it was rebound above
    import scripts.poc_extract_credit_factors as pc
    print("\n=== TEST START: user_733 ===")
    print(f"1. Opening PDF: {pdf}")
    print(f"2. Calling detection function...")
    res = pc.combined_sample_color_for_phrase(doc, '1 Inq Last 4 Mo', expected_color=None, page_limit=1)
    print(f"3. Detection result: {res}")
    assert res is not None, 'Phrase not found for user_733'
    pidx, text, hexv, rgb, bbox, method = res
    print(f"4. Checking color: {rgb}")
    cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
    print(f"5. Detected category: {cat}, method: {method}")
    assert cat == 'red', f'Expected red for \"Ing Last 4 Mo\" but got {cat} (method={method})'
