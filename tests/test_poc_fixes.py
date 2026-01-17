import pytest
from pathlib import Path
from tests.test_pdf_extraction_ground_truth import load_json, compare_dicts
from src.scripts.pdf_color_extraction import extract_pdf_all_fields


@pytest.mark.importorskip('fitz')
def test_user_582_matches_ground_truth():
    pdf = 'data/pdf_analysis/user_582_credit_summary_2025-09-01_100800.pdf'
    assert Path(pdf).exists()
    extracted = extract_pdf_all_fields(pdf)
    gt = load_json(Path('data/extracted') / (Path(pdf).stem + '_ground_truth.json'))
    assert compare_dicts(extracted, gt), 'user_582 mismatch'


@pytest.mark.importorskip('fitz')
def test_user_1314_drop_bad_auth_preserved():
    """Regression: ensure 'Drop Bad Auth' summary factor present for user_1314"""
    pdf = 'data/pdf_analysis/user_1314_credit_summary_2025-09-01_092724.pdf'
    assert Path(pdf).exists()
    rec = extract_pdf_all_fields(pdf)
    assert any('Drop Bad Auth' in f.get('factor','') for f in rec.get('credit_factors', [])), 'Drop Bad Auth factor was dropped'


@pytest.mark.importorskip('fitz')
def test_user_1254_matches_ground_truth():
    pdf = 'data/pdf_analysis/user_1254_credit_summary_2025-09-01_095528.pdf'
    assert Path(pdf).exists()
    extracted = extract_pdf_all_fields(pdf)
    gt_path = Path('data/extracted') / (Path(pdf).stem + '_ground_truth.json')
    if not gt_path.exists():
        # Ground truth missing for this sample; save unvalidated and xfail
        import json
        with open(gt_path.parent / (gt_path.stem + '_ground_truth_unvalidated.json'), 'w', encoding='utf-8') as fh:
            json.dump(extracted, fh, indent=2, ensure_ascii=False)
        pytest.xfail('Ground truth missing for user_1254; extraction saved as unvalidated')
    gt = load_json(gt_path)
    assert compare_dicts(extracted, gt), 'user_1254 mismatch'


@pytest.mark.importorskip('fitz')
def test_user_1514_matches_ground_truth():
    pdf = 'data/pdf_analysis/user_1514_credit_summary_2025-09-01_145557.pdf'
    assert Path(pdf).exists()
    extracted = extract_pdf_all_fields(pdf)
    gt_path = Path('data/extracted') / (Path(pdf).stem + '_ground_truth.json')
    if not gt_path.exists():
        import json
        with open(gt_path.parent / (gt_path.stem + '_ground_truth_unvalidated.json'), 'w', encoding='utf-8') as fh:
            json.dump(extracted, fh, indent=2, ensure_ascii=False)
        pytest.xfail('Ground truth missing for user_1514; extraction saved as unvalidated')
    gt = load_json(gt_path)
    assert compare_dicts(extracted, gt), 'user_1514 mismatch'
