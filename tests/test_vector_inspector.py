import pytest
from pathlib import Path
pytest.importorskip('fitz')
import fitz
from scripts.poc_extract_credit_factors import load_expectations_from_dir, map_color_to_cat, span_color_hex, ROOT

PDF_DIR = ROOT / 'data' / 'pdf_analysis'


def normalize_phrase(s: str) -> str:
    return ' '.join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def test_span_colors_detected_for_1314():
    ex = load_expectations_from_dir(ROOT / 'data' / 'pdf_analysis')
    fname = None
    for f in ex:
        if '1314' in f:
            fname = f
            break
    assert fname, '1314 expectation file not found'
    expects = ex[fname]
    # pick a phrase we expect to have a green marker
    target_phrase = None
    for ph, col in expects.items():
        if col == 'green':
            target_phrase = ph
            break
    assert target_phrase, 'No green expectation phrase present in 1314 expectations'
    pdf_path = PDF_DIR / fname
    assert pdf_path.exists(), f'pdf missing: {pdf_path}'
    doc = fitz.open(str(pdf_path))
    found = False
    for p in range(len(doc)):
        td = doc.load_page(p).get_text('dict')
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if not text:
                    continue
                if target_phrase.lower() in text.lower() or text.lower() in target_phrase.lower():
                    hexv, rgb = span_color_hex(ln.get('spans', []))
                    assert rgb is not None, f'No span color found for phrase: {target_phrase}'
                    assert map_color_to_cat(rgb) == 'green', f'Expected green but got {map_color_to_cat(rgb)} for {target_phrase}'
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found, f'Phrase not located in PDF: {target_phrase}'