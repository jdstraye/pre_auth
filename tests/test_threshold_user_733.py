import fitz
import importlib
import src.scripts.pdf_color_extraction as pc


def test_user_733_best_red_score_and_threshold():
    pdf = 'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
    doc = fitz.open(pdf)
    # reload module to ensure helpers are bound (handles import-order races)
    importlib.reload(pc)
    assert hasattr(pc, 'get_candidates_for_phrase'), 'get_candidates_for_phrase missing after reload'
    cands = pc.get_candidates_for_phrase(doc, 'Ing Last 4 Mo', page_limit=1)
    assert cands, 'No candidates found for user_733'
    red_scores = [c['score'] for c in cands if pc.map_color_to_cat(c['rgb']) == 'red']
    assert red_scores, 'No red candidates found for user_733'
    best_red = max(red_scores)
    # sanity: best_red should be > 0.5 (span-first strong match expected)
    assert best_red > 0.5

    # Test impl behavior at boundary: setting MIN_ACCEPT_SCORE to best_red should still accept
    old = getattr(pc, 'MIN_ACCEPT_SCORE', None)
    try:
        pc.MIN_ACCEPT_SCORE = best_red
        res = pc.combined_sample_color_for_phrase_impl(doc, 'Ing Last 4 Mo', page_limit=1)
        assert res is not None, 'Impl failed to return a candidate at threshold'
        _, _, _, rgb, _, method = res
        assert pc.map_color_to_cat(rgb) == 'red', f'Expected red at MIN_ACCEPT_SCORE={best_red} but got {pc.map_color_to_cat(rgb)} (method={method})'

        # Slightly above should cause impl to reject this candidate
        pc.MIN_ACCEPT_SCORE = best_red + 1e-6
        res2 = pc.combined_sample_color_for_phrase_impl(doc, 'Ing Last 4 Mo', page_limit=1)
        # impl may fallback to color_first or None; ensure it does not return a red span at or above stricter threshold
        if res2:
            _, _, _, rgb2, _, method2 = res2
            assert pc.map_color_to_cat(rgb2) != 'red' or method2 != 'spans', 'Impl still returned red span above threshold'
    finally:
        if old is not None:
            pc.MIN_ACCEPT_SCORE = old
        else:
            delattr(pc, 'MIN_ACCEPT_SCORE')
