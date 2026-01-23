import importlib, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pymupdf_compat import fitz
pc = importlib.import_module('scripts.poc_extract_credit_factors')
# prefer the public delegator, fall back to impl if present
sample_color_from_glyphs = getattr(pc, 'sample_color_from_glyphs', getattr(pc, 'sample_color_from_glyphs_impl', None))
combined_sample_color_for_phrase = getattr(pc, 'combined_sample_color_for_phrase')
map_color_to_cat = getattr(pc, 'map_color_to_cat')


def _find_spans_for_text(page, target):
    td = page.get_text('dict')
    for b in td.get('blocks', []):
        for ln in b.get('lines', []):
            text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
            if not text:
                continue
            if target.lower() in text.lower():
                return ln.get('spans', []), text
    return None, None


def test_sample_color_from_glyphs_user_733():
    pdf = 'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
    doc = fitz.open(pdf)
    page = doc.load_page(0)
    spans, text = _find_spans_for_text(page, '1 Inq Last 4 Mo')
    assert spans is not None, 'Expected to find corrected phrase spans on the page'
    hexv, rgb, conf = sample_color_from_glyphs(page, spans, scale=3)
    assert rgb is not None and conf > 0.2, f'Expected glyph sampling to find a color, got {hexv} conf={conf}'
    cat = map_color_to_cat(rgb)
    assert cat == 'red', f'Expected red for the correct span but got {cat}'


def test_combined_sampler_recovers_ocrish_phrase():
    pdf = 'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
    doc = fitz.open(pdf)
    res = combined_sample_color_for_phrase(doc, 'Ing Last 4 Mo', page_limit=1)
    assert res is not None, 'Phrase not found for user_733'
    pidx, text, hexv, rgb, bbox, method = res
    cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
    assert cat == 'red', f'Expected red for "Ing Last 4 Mo" but got {cat} (method={method})'
