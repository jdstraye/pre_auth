import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pymupdf_compat import fitz
from scripts.poc_extract_credit_factors import sample_color_from_glyphs, combined_sample_color_for_phrase, map_color_to_cat

pdf = 'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf'
doc = fitz.open(pdf)
page = doc.load_page(0)
# find corrected phrase spans
spans = None
for b in page.get_text('dict').get('blocks',[]):
    for ln in b.get('lines',[]):
        text = ''.join([s.get('text','') for s in ln.get('spans',[])]).strip()
        if '1 Inq Last 4 Mo' in text:
            spans = ln.get('spans', [])
            print('found text:', text)
            break
    if spans:
        break
if spans is None:
    print('could not find corrected phrase spans on page')
else:
    h, rgb, conf = sample_color_from_glyphs(page, spans, scale=3)
    print('glyph sample ->', h, rgb, conf, 'cat=', map_color_to_cat(rgb) if rgb else None)

# now call combined sampler on OCRish phrase
res = combined_sample_color_for_phrase(doc, 'Ing Last 4 Mo', page_limit=1)
print('combined sampler res:', res)
if res:
    pidx, text, hexv, rgb, bbox, method = res
    print('cat=', map_color_to_cat(rgb) if rgb else None, 'method=', method)
