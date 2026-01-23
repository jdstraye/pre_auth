import csv, sys, os, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pc = importlib.import_module('src.scripts.pdf_color_extraction')
import fitz

TEST_CASES = [
    ('data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf', 'Ing Last 4 Mo', 'red'),
]

OUT = 'output/color_sampler_report.csv'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pdf','phrase','expected','page','text','hex','rgb','method','cat','conf'])
    for pdf, phrase, expected in TEST_CASES:
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            w.writerow([pdf, phrase, expected, 'ERROR', str(e), '', '', '', '', ''])
            continue
        try:
            res = pc.combined_sample_color_for_phrase(doc, phrase, page_limit=1)
            if res:
                pidx, text, hexv, rgb, bbox, method = res
                cat = pc.map_color_to_cat(rgb) if rgb else 'neutral'
                conf = ''
                # attempt to get glyph confidence if available
                try:
                    sfunc = getattr(pc, 'sample_color_from_glyphs_impl', None) or getattr(pc, 'sample_color_from_glyphs', None)
                    if sfunc and method in ('glyph_cluster',):
                        # try approximate spans from the found bbox for confidence estimate
                        try:
                            page = doc.load_page(pidx)
                            td = page.get_text('dict')
                            spans = []
                            for b in td.get('blocks', []):
                                for ln in b.get('lines', []):
                                    textln = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                                    if textln and text in textln:
                                        spans = ln.get('spans', [])
                                        break
                                if spans:
                                    break
                            _, _rgb, conf = sfunc(page, spans)
                        except Exception:
                            conf = ''
                except Exception:
                    conf = ''
                w.writerow([pdf, phrase, expected, pidx, text, hexv, rgb, method, cat, conf])
            else:
                w.writerow([pdf, phrase, expected, 'NOT_FOUND', '', '', '', '', '', ''])
        except Exception as e:
            w.writerow([pdf, phrase, expected, 'ERROR', str(e), '', '', '', '', ''])
print('wrote', OUT)
