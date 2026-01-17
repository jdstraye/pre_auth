import json
from pathlib import Path

from src.scripts.pdf_color_extraction import extract_pdf_all_fields


def test_regression_snapshot_for_user_1314(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    pdf = repo_root / "data" / "pdf_analysis" / "user_1314.pdf"
    assert pdf.exists(), "fixture pdf for user_1314 must exist in data/pdf_analysis"

    rec = extract_pdf_all_fields(str(pdf))
    # smoke: ensure we have candidates and candidate fields
    cands = rec.get("candidates") or rec.get("credit_factors")
    assert cands and len(cands) > 0
    first = cands[0]
    # extractor historically used different keys for the line text ("factor", "text", "line_text"); accept any
    assert any(k in first for k in ("line_text", "text", "factor"))
    assert "page" in first
    assert "spans" in first

    # exercise regen tool via import (not CLI) to ensure output shape
    from scripts.regen_regression_set import candidate_snapshot_for_pdf

    snap = candidate_snapshot_for_pdf(pdf, seed_from_poc=True)
    assert snap["pdf_id"] == pdf.stem
    assert "candidates" in snap
    assert isinstance(snap["candidates"], list)
    # ensure poc_candidates key exists when seed_from_poc=True
    assert "poc_candidates" in snap
    # verify candidate shape
    for c in snap["candidates"][:3]:
        assert "candidate_id" in c and "line_text" in c and "spans" in c
