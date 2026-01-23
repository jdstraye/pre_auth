import pytest
from pathlib import Path
from src.scripts.pdf_color_extraction import extract_pdf_all_fields
from tests.test_pdf_extraction_ground_truth import load_json, compare_dicts

@pytest.mark.parametrize("uid,dt", [
    ("1140", "132703"),
    ("1314", "092724"),
])
def test_address_integration_matches_ground_truth(uid, dt):
    pdf_path = Path(f"data/pdf_analysis/user_{uid}_credit_summary_2025-09-01_{dt}.pdf")
    gt_path = Path("data/extracted") / f"user_{uid}_credit_summary_2025-09-01_{dt}_ground_truth.json"
    gt = load_json(gt_path)
    extracted = extract_pdf_all_fields(str(pdf_path))
    # Only assert address compatibility to lock-in end-to-end behavior
    assert compare_dicts({'address': extracted.get('address')}, {'address': gt.get('address')})
