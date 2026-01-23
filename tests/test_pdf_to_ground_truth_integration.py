import json
from pathlib import Path
import tempfile
import pytest

from scripts import pdf_to_ground_truth as pt


def test_pdf_to_ground_truth_integration(tmp_path):
    # Skip test if PyMuPDF is not available
    pytest.importorskip('fitz')
    # Find a sample pdf in data/pdf_analysis
    d = Path('data/pdf_analysis')
    candidates = list(d.glob('user_1314_*.pdf'))
    if not candidates:
        pytest.skip('no sample PDF found for user_1314')
    pdf = candidates[0]
    outp = tmp_path / (pdf.stem + '_ground_truth_unvalidated.json')
    # Run the script main to generate GT + include-spans
    ret = pt.main([str(pdf), '--include-spans', '--out', str(outp)])
    assert outp.exists()
    # check enriched .with_spans.json exists after attach_spans_to_gt
    enriched = outp.with_name(outp.stem + '.with_spans.json')
    assert enriched.exists()
    data = json.loads(enriched.read_text())
    # At least one factor should have page/bbox/spans and canonical_key
    factors = data.get('credit_factors', [])
    assert factors, 'no credit_factors found in enriched GT'
    found = False
    for f in factors:
        if f.get('page') is not None or f.get('bbox') is not None or f.get('spans'):
            found = True
            break
    assert found, 'no spans/bbox/page attached to any factor'
    # canonical_key should be present on at least one factor
    assert any(f.get('canonical_key') for f in factors), 'no canonical_key set on any factor'
    # Also ensure top-level annotated fields are present for training: credit_score and credit_card_open_totals
    assert data.get('credit_score') is not None
    assert (data.get('credit_score_bbox') is not None) or (data.get('credit_score_spans') is not None) or (data.get('credit_score_color') is not None)
    cc = data.get('credit_card_open_totals')
    assert cc is not None
    assert cc.get('hex') or cc.get('color') is not None
    # If include_spans attached, expect page/bbox or spans
    assert any(k in cc for k in ('bbox','page','spans'))