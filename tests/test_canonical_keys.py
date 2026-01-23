import pytest
from src.scripts.pdf_color_extraction import extract_pdf_all_fields
from pathlib import Path

PDF = 'data/pdf_analysis/user_582_credit_summary_2025-09-01_100800.pdf'


def test_extractor_emits_canonical_keys():
    rec = extract_pdf_all_fields(PDF)
    # inquiries canonical
    assert 'inquiries_last_6_months' in rec
    assert isinstance(rec['inquiries_last_6_months'], int)
    assert 'inquiries_6mo' not in rec
    # collections canonical
    assert 'collections_open' in rec and isinstance(rec['collections_open'], int)
    assert 'collections_closed' in rec and isinstance(rec['collections_closed'], int)
    # address canonical: list
    assert 'address' in rec
    assert isinstance(rec['address'], list)
    # credit_card_open_totals canonical structure if present
    cct = rec.get('credit_card_open_totals')
    if cct is not None:
        assert isinstance(cct, dict)
        for k in ('color','balance','limit','Percent','Payment'):
            assert k in cct
    # credit_score flat
    assert isinstance(rec.get('credit_score'), (int, type(None)))
    assert 'credit_score_color' in rec
