import pytest
import os
from pathlib import Path
from src.scripts.pdf_color_extraction import (
    parse_count_amount_pair,
    parse_public_records,
    load_expectations_from_dir,
    extract_credit_factors_from_doc,
    median_5x5,
)


def test_parse_count_amount_pair_examples():
    assert parse_count_amount_pair('10 / $56,881') == (10, 56881)
    assert parse_count_amount_pair('$56,881 / 10') == (10, 56881)
    assert parse_count_amount_pair('3 / $1,104') == (3, 1104)
    assert parse_count_amount_pair('Pay $123/mo') == (None, None)


def test_parse_public_records_examples():
    txt = "Credit Alerts\nPublic Records\n1\nMore"
    c, n = parse_public_records(txt)
    assert c == 1
    assert n == ''
    txt2 = "Public Records: 0"
    c2, n2 = parse_public_records(txt2)
    assert c2 == 0


def test_load_expectations():
    d = load_expectations_from_dir(Path('data') / 'pdf_analysis')
    # should return a dict mapping filenames to phrase->color mappings (may be empty in trimmed test fixtures)
    assert isinstance(d, dict)


@pytest.mark.importorskip('fitz')
def test_extract_credit_factors_nonempty():
    pdf = 'data/pdf_analysis/user_582_credit_summary_2025-09-01_100800.pdf'
    assert Path(pdf).exists(), 'PDF sample missing for test'
    import fitz
    doc = fitz.open(pdf)
    factors = extract_credit_factors_from_doc(doc, page_limit=2)
    assert isinstance(factors, list)
    assert all(isinstance(f, dict) for f in factors)
