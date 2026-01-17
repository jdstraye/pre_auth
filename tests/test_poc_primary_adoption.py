import pytest
from pathlib import Path

from src.scripts import pdf_color_extraction as pdf_mod


@pytest.mark.importorskip('fitz')
def test_poc_preferred_over_fallback(monkeypatch):
    pdf = Path('data/pdf_analysis/user_582_credit_summary_2025-09-01_100800.pdf')
    assert pdf.exists(), "Sample PDF missing for test"

    def stub(doc, page_limit=None):
        # Return a distinctive marker factor so we can assert it propagated
        return [{'factor': 'POC_PRIMARY_MARKER', 'color': 'red'}]

    monkeypatch.setattr(pdf_mod, 'extract_credit_factors_from_doc', stub)
    rec = pdf_mod.extract_pdf_all_fields(str(pdf))
    assert isinstance(rec.get('credit_factors'), list)
    assert rec['credit_factors'][0]['factor'] == 'POC_PRIMARY_MARKER'
    # counts should reflect the stubbed red factor
    assert rec.get('red_credit_factors_count', 0) >= 1


@pytest.mark.importorskip('fitz')
def test_fallback_used_when_poc_empty(monkeypatch):
    pdf = Path('data/pdf_analysis/user_582_credit_summary_2025-09-01_100800.pdf')
    assert pdf.exists(), "Sample PDF missing for test"

    # Make POC return empty list; extractor must still produce credit_factors via fallback heuristics
    monkeypatch.setattr(pdf_mod, 'extract_credit_factors_from_doc', lambda doc, page_limit=None: [])
    rec = pdf_mod.extract_pdf_all_fields(str(pdf))
    assert isinstance(rec.get('credit_factors'), list)
    # Expect fallback to find at least one factor for this sample PDF
    assert len(rec['credit_factors']) > 0
