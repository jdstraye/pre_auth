import pytest
from scripts import rebuild_script_from_spans as r


def test_choose_color_from_spans_green_prefers_colored():
    # one colored token and one white token -> should pick green
    rgbs = [[116, 172, 125], [255, 255, 255]]
    hexv, cat = r.choose_color_from_spans(rgbs)
    assert cat == 'green'
    assert hexv is not None and hexv.startswith('#')


def test_choose_color_from_spans_pale_gray_becomes_neutral():
    # pale gray should not be mapped to a colored category
    rgbs = [[189, 190, 192]]
    hexv, cat = r.choose_color_from_spans(rgbs)
    assert cat == 'neutral'
    assert hexv is None


@pytest.mark.parametrize('rgb,expected', [
    ((116, 172, 125), 'green'),
    ((220, 53, 69), 'red'),
    ((190, 190, 190), 'neutral'),
])
def test_map_color_simple(rgb, expected):
    assert r.map_color_simple(rgb) == expected


def test_sample_phrase_bbox_from_pdf_existing_pdf():
    # Use a real sample from the repo where we know the phrase color (user_1314 $16320 Unpaid Collection(s))
    stem = 'user_1314_credit_summary_2025-09-01_092724'
    phrase = '$16320 Unpaid Collection(s)'
    hexv, rgb = r._sample_phrase_bbox_from_pdf(stem, phrase)
    assert hexv is not None
    assert isinstance(rgb, tuple) and len(rgb) == 3
    # sampled color should map to green for this known example
    assert r.map_color_simple(rgb) == 'green'
