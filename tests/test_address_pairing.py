import pytest
from src.scripts.pdf_color_extraction import pair_addresses_from_candidates


def test_pairing_with_global_city_line():
    candidates = [
        "846 Northluke 1604 W, San Antonio Te, TX 78248",
        "14013 Hadley Run",
    ]
    all_lines = [
        "Some header",
        "LIVE OAK, TX. 78233",
        "14013 Hadley Run, Live Oak, TX. 78233",
    ]
    out = pair_addresses_from_candidates(candidates, all_lines)
    assert isinstance(out, list)
    assert any('Live Oak' in a for a in out)


def test_single_street_with_city_in_candidates():
    candidates = [
        "123 Maple Ave",
        "Anytown, NY 12345",
    ]
    all_lines = candidates[:]
    out = pair_addresses_from_candidates(candidates, all_lines)
    assert isinstance(out, str)
    assert 'Anytown' in out


def test_no_street_candidates_returns_none():
    candidates = ["No address here", "Something else"]
    all_lines = candidates[:]
    out = pair_addresses_from_candidates(candidates, all_lines)
    assert out is None


def test_pairing_with_misaligned_inline_city():
    """Ensure that when one candidate includes an inline city/zip (possibly malformed) and the correct city appears elsewhere,
    the helper pairs the appropriate city with the matching street (e.g., Live Oak for the second street)."""
    candidates = [
        "846 Northluke 1604 W, San Antonio Te, TX 78248",
        "14013 Hadley Run",
    ]
    all_lines = [
        "Some header",
        "LIVE OAK, TX. 78233",
        "846 Northluke 1604 W, San Antonio Te, TX 78248",
        "14013 Hadley Run, Live Oak, TX. 78233",
    ]
    out = pair_addresses_from_candidates(candidates, all_lines)
    assert isinstance(out, list)
    # second address must be paired with Live Oak
    assert any('Live Oak' in a for a in out)
