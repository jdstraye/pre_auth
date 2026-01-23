import csv, sys, os, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pc = importlib.import_module('src.scripts.pdf_color_extraction')
import fitz

# choose 100 PDFs
pdfs = sorted([p for p in os.listdir('data/pdf_analysis') if p.endswith('.pdf')])[:100]
phrases = ['Ing Last 4 Mo']
OUT = 'output/color_sampler_threshold_scan.csv'
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pdf','phrase','page','text','method','hex','rgb','cat','score','reason'])
    for pdf in pdfs:
        pdf_path = os.path.join('data/pdf_analysis', pdf)
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            w.writerow([pdf, '', 'ERROR', str(e), '', '', '', '', ''])
            continue
        for phrase in phrases:
            try:
                res = pc.combined_sample_color_for_phrase(doc, phrase, page_limit=1)
                if not res:
                    w.writerow([pdf, phrase, 'NOT_FOUND', '', '', '', '', '', ''])
                    continue
                pidx, text, hexv, rgb, bbox, method = res
                cat = pc.map_color_to_cat(rgb) if rgb else 'neutral'
                # estimate color_conf and compute score locally (avoid depending on internal helpers)
                color_conf = 0.0
                if method == 'spans':
                    color_conf = 0.95
                elif method == 'glyph_cluster':
                    color_conf = 0.5
                elif method == 'bbox':
                    color_conf = 0.5
                elif method in ('pix_median','color_first'):
                    color_conf = 0.4
                # text_sim
                import re, difflib
                def _norm(s):
                    return ' '.join(re.findall(r"\w+", (s or '').lower()))
                tscore = difflib.SequenceMatcher(None, _norm(phrase), _norm(text)).ratio()
                # local scoring with red bonus and numeric penalty
                def is_pure_numeric(s):
                    import re
                    return bool(re.fullmatch(r"[\d\s\$\,\/\.\%\-]+", (s or '').strip()))
                def is_red_color(rgb):
                    try:
                        return pc.map_color_to_cat(rgb) == 'red'
                    except Exception:
                        return False
                w_text=0.5; w_color_conf=0.3; red_bonus=0.2; numeric_penalty=0.1
                bonus = red_bonus if is_red_color(rgb) else 0.0
                penalty = numeric_penalty if is_pure_numeric(text) else 0.0
                score = max(0.0, min(1.0, w_text*tscore + w_color_conf*color_conf + bonus - penalty))
                # Rejection reason: include a short message for why candidate was rejected (empty if accepted)
                reason = ''
                if score < pc.MIN_ACCEPT_SCORE:
                    reason = f"Score {score:.3f} < {pc.MIN_ACCEPT_SCORE:.3f}"
                w.writerow([pdf, phrase, pidx, text, method, hexv, rgb, cat, score, reason])
            except Exception as e:
                w.writerow([pdf, phrase, 'ERROR', str(e), '', '', '', '', ''])
print('wrote', OUT)
