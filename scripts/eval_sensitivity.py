import csv, sys, os, importlib, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pc = importlib.import_module('src.scripts.pdf_color_extraction')
import fitz

# sample 100 PDFs
pdfs = sorted([p for p in os.listdir('data/pdf_analysis') if p.endswith('.pdf')])
random.seed(42)
sample = [os.path.join('data/pdf_analysis', p) for p in random.sample(pdfs, min(100, len(pdfs)))]
OUT = 'output/color_sampler_candidates.csv'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pdf','phrase','p','text','method','hex','rgb','text_sim','color_conf','score','cat'])
    for pdf in sample:
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            w.writerow([pdf,'','ERROR',str(e),'','','','','','',''])
            continue
        # use a set of general phrases to exercise sampler
        phrases = ['Credit Factors','Total Rev Usage','Inquires Last 4 Mo','1 Inq Last 4 Mo','Ing Last 4 Mo','Rev Accounts (Open)']
        for phrase in phrases:
            try:
                cands = pc.get_candidates_for_phrase(doc, phrase, page_limit=1)
                for c in cands[:5]:
                    cat = pc.map_color_to_cat(c.get('rgb')) if c.get('rgb') else 'neutral'
                    w.writerow([pdf, phrase, c.get('p'), c.get('text'), c.get('method'), c.get('hex'), c.get('rgb'), c.get('text_sim'), c.get('color_conf'), c.get('score'), cat])
            except Exception as e:
                w.writerow([pdf, phrase, 'ERROR', str(e),'','','','','','',''])
print('wrote candidates csv ->', OUT)
