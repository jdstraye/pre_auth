#!/usr/bin/env python3
"""Run a small extraction on a set of PDFs and write per-PDF JSON outputs.
Usage: python scripts/run_sample_extraction.py user_1314,user_2095,user_2096
"""
import json, sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from src.scripts.pdf_color_extraction import extract_pdf_all_fields, parse_count_amount_pair, map_line_to_canonical, span_color_hex, rgb_to_hex_tuple, map_color_to_cat, combined_sample_color_for_phrase, parse_public_records, normalize_factors
try:
    import fitz
except Exception:
    print('Missing dependency: PyMuPDF (fitz) not available. Install with: pip install pymupdf')
    raise
ROOT = Path('.').resolve()
PDF_DIR = ROOT / 'data' / 'pdf_analysis'
OUT = ROOT / 'data' / 'extracted'
OUT.mkdir(parents=True, exist_ok=True)

import random

# If no args provided, pick 3 random PDFs from PDF_DIR
if len(sys.argv) < 2:
    all_pdfs = [p.name for p in PDF_DIR.iterdir() if p.suffix.lower() == '.pdf']
    if not all_pdfs:
        print('No PDFs found in', PDF_DIR)
        sys.exit(1)
    sample = random.sample(all_pdfs, min(3, len(all_pdfs)))
    print('No PDFs provided; selecting random sample:', sample)
    pdfs = sample
else:
    pdfs = sys.argv[1].split(',')
for pbase in pdfs:
    pdf_path = PDF_DIR / pbase
    if not pdf_path.exists():
        # try with pdf suffix
        pdf_path = PDF_DIR / f"{pbase}.pdf"
    if not pdf_path.exists():
        # Try to find URL in data/prefi_report-20251218-slack1409.json and download
        import json, requests
        report = JSON = Path('data/prefi_report-20251218-slack1409.json')
        if report.exists():
            with open(report) as fh:
                data = json.load(fh)
            found_url = None
            for rec in data.get('data', []):
                url = rec.get('credit_summary_pdf_url')
                if url and url.endswith(str(pbase)):
                    found_url = url
                    break
            if not found_url:
                # also try suffix match
                for rec in data.get('data', []):
                    url = rec.get('credit_summary_pdf_url')
                    if url and pbase in url:
                        found_url = url
                        break
            if found_url:
                print('Downloading', found_url)
                try:
                    r = requests.get(found_url, timeout=20)
                    if r.status_code == 200:
                        pdf_path.write_bytes(r.content)
                        print('Saved', pdf_path)
                    else:
                        print('Download failed:', r.status_code)
                except Exception as e:
                    print('Download error:', e)
        if not pdf_path.exists():
            print('PDF not found:', pbase)
            continue
    print('Processing', pdf_path)
    doc = fitz.open(str(pdf_path))
    lines = []
    page_pivots = {}
    for p in range(len(doc)):
        page = doc.load_page(p)
        td = page.get_text('dict')
        # collect x0 positions to estimate a left/right pivot
        x0s = []
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                for s in ln.get('spans', []):
                    bbox = s.get('bbox')
                    if bbox:
                        x0s.append(bbox[0])
        if x0s:
            page_pivots[p] = (min(x0s) + max(x0s)) / 2.0
        else:
            page_pivots[p] = None
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if not text:
                    continue
                # get the x0 of the first span (fallback to pivot classification)
                spans = ln.get('spans', [])
                x0 = None
                if spans:
                    bbox = spans[0].get('bbox')
                    if bbox:
                        x0 = bbox[0]
                lines.append((p, text, spans, x0))
    full_text = '\n'.join([doc.load_page(i).get_text() for i in range(len(doc))])
    # Use canonical extractor to create the authoritative record
    rec = extract_pdf_all_fields(str(pdf_path))
    # Ensure backward-compatible keys and metadata
    rec['filename'] = rec.get('filename', pdf_path.name)
    rec['source'] = rec.get('source', str(pdf_path))
    for _k in ['installment_open_count','installment_open_total','revolving_open_count','revolving_open_total']:
        rec[_k] = rec.get(_k, '')
    rec['public_record_note'] = rec.get('public_record_note', '')

    # normalize address if present with multiple lines or left/right lists

    # normalize address if present with multiple lines or left/right lists
    if rec.get('address'):
        addr = rec['address']
        import re
        def _pick_addr(a):
            parts = [p.strip() for p in a.splitlines() if p.strip()]
            chosen = None
            for p in reversed(parts):
                if re.search(r",\s*[A-Z]{2}\.\?\s*\d{5}", p):
                    chosen = p
                    break
            if not chosen and parts:
                chosen = parts[-1]
            return chosen or a
        if isinstance(addr, list):
            normalized = [_pick_addr(a) if isinstance(a, str) else a for a in addr]
            # if both sides normalize to the same string, collapse to single string
            if len(normalized) == 2 and normalized[0] == normalized[1]:
                rec['address'] = normalized[0]
            else:
                rec['address'] = normalized
        else:
            rec['address'] = _pick_addr(addr)

    # Fallback scanning for installment/revolving counts when missing (populate legacy wide_row)
    if not rec.get('installment_open_count') or rec.get('installment_open_count') == '' or rec.get('installment_open_count') == 0:
        prev_ln = ''
        for p in range(len(doc)):
            td = doc.load_page(p).get_text('dict')
            for b in td.get('blocks', []):
                for ln in b.get('lines', []):
                    line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                    if not line_text:
                        prev_ln = ''
                        continue
                    combined = (prev_ln + ' ' + line_text).strip()
                    cnt, amt = parse_count_amount_pair(combined)
                    if cnt is None and amt is None:
                        cnt, amt = parse_count_amount_pair(line_text)
                    if (cnt is not None or amt is not None) and 'install' in combined.lower():
                        if cnt is not None:
                            rec['installment_open_count'] = int(cnt)
                        if amt is not None:
                            rec['installment_open_total'] = int(amt)
                        break
                    prev_ln = line_text
                if rec.get('installment_open_count'):
                    break
            if rec.get('installment_open_count'):
                break

    outp = OUT / f"{pdf_path.name}.json"
    # Provide a legacy 'wide_row' key for older consumers that expect scaffolded numeric parsing results
    wide_row = {
        'installment_open_count': rec.get('installment_open_count', ''),
        'installment_open_total': rec.get('installment_open_total', '')
    }
    with open(outp,'w') as fh:
        json.dump({'rec': rec, 'wide_row': wide_row}, fh, indent=2)
    print('Wrote', outp)
print('Done')
