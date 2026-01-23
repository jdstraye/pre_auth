import sys, os, importlib
sys.path.insert(0, os.path.abspath('..'))
pc = importlib.import_module('scripts.poc_extract_credit_factors')
from src.pymupdf_compat import fitz
print('has impl?', hasattr(npc, 'combined_sample_color_for_phrase_impl'))
print('has glyph impl?', hasattr(npc, 'sample_color_from_glyphs_impl'))
pdf='data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
doc = fitz.open(pdf)
if hasattr(npc, 'combined_sample_color_for_phrase_impl'):
    print('calling impl on OCRish phrase...')
    r = npc.combined_sample_color_for_phrase_impl(doc, 'Ing Last 4 Mo', page_limit=1)
    print('impl result:', r)
    print('calling impl on corrected phrase...')
    r2 = npc.combined_sample_color_for_phrase_impl(doc, '1 Inq Last 4 Mo', page_limit=1)
    print('impl result corrected:', r2)
    if r2:
        pidx, text, hexv, rgb, bbox, method = r2
        page = doc.load_page(pidx)
        # try to find spans for the exact text
        for b in page.get_text('dict').get('blocks', []):
            for ln in b.get('lines', []):
                t = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if '1 Inq Last 4 Mo' in t:
                    spans = ln.get('spans', [])
                    print('found spans text=', t)
                    ghex, grgb, gconf = npc.sample_color_from_glyphs_impl(page, spans)
                    print('glyph sample on corrected:', ghex, grgb, gconf)
                    break
            else:
                continue
            break
else:
    print('impl missing')
