#!/usr/bin/env python3
"""Run a small extraction on a set of PDFs and write per-PDF JSON outputs.
Usage: python scripts/run_sample_extraction.py user_1314,user_2095,user_2096
"""
import json, sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from scripts.poc_extract_credit_factors import extract_record_level, parse_count_amount_pair, map_line_to_canonical, span_color_hex, rgb_to_hex_tuple, map_color_to_cat, combined_sample_color_for_phrase, parse_public_records, normalize_factors
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
    rec = extract_record_level(full_text)
    # OCR-based fallback for numeric pairs
    try:
        # iterate by index so we can look ahead/back with spatial awareness
        for idx in range(len(lines)):
            p_idx, ln_text, sp, x0 = lines[idx]
            ln_low = ln_text.lower()
            pivot = page_pivots.get(p_idx)
            # if this looks like a left-column label, look ahead for right-column numeric
            if pivot is not None and x0 is not None and x0 <= pivot - 10:
                # scan the next few lines on the same page for a right-column numeric token
                for j in range(idx+1, min(len(lines), idx+6)):
                    p2, ln2_text, sp2, x2 = lines[j]
                    if p2 != p_idx:
                        break
                    if x2 is None:
                        continue
                    # right-column candidate (use pivot tolerance to catch right column even when pivot shifts)
                    if x2 > pivot - 10:
                        cnt, amt = parse_count_amount_pair(ln2_text)
                        if cnt is None and amt is None:
                            # try combining left label + right text
                            cnt, amt = parse_count_amount_pair(ln_text + ' ' + ln2_text)
                        if cnt is not None or amt is not None:
                            if 'install' in ln_low:
                                if rec.get('installment_open_count') is None and cnt is not None:
                                    rec['installment_open_count'] = int(cnt)
                                if rec.get('installment_open_total') is None and amt is not None:
                                    rec['installment_open_total'] = int(amt)
                            if 'revolv' in ln_low or 'rev' in ln_low:
                                if rec.get('revolving_open_count') is None and cnt is not None:
                                    rec['revolving_open_count'] = int(cnt)
                                if rec.get('revolving_open_total') is None and amt is not None:
                                    rec['revolving_open_total'] = int(amt)
                            break
            # fallback: if this line itself contains a count/amount pair, assign appropriately
            cnt0, amt0 = parse_count_amount_pair(ln_text)
            if (cnt0 is not None or amt0 is not None):
                if 'install' in ln_low:
                    if rec.get('installment_open_count') is None and cnt0 is not None:
                        rec['installment_open_count'] = int(cnt0)
                    if rec.get('installment_open_total') is None and amt0 is not None:
                        rec['installment_open_total'] = int(amt0)
                if 'revolv' in ln_low or 'rev' in ln_low:
                    if rec.get('revolving_open_count') is None and cnt0 is not None:
                        rec['revolving_open_count'] = int(cnt0)
                    if rec.get('revolving_open_total') is None and amt0 is not None:
                        rec['revolving_open_total'] = int(amt0)
    except Exception:
        pass

    # If still missing installment/revolving info, try OCR on rendered pages (pytesseract)
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        # only run if keys missing
        if rec.get('installment_open_count') is None or rec.get('installment_open_total') is None or rec.get('revolving_open_count') is None or rec.get('revolving_open_total') is None:
            for p in range(len(doc)):
                pix = doc.load_page(p).get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
                img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                for oline in ocr_text.splitlines():
                    if not oline.strip():
                        continue
                    cnt, amt = parse_count_amount_pair(oline)
                    if (cnt is not None or amt is not None):
                        low = oline.lower()
                        if 'install' in low:
                            if rec.get('installment_open_count') is None and cnt is not None:
                                rec['installment_open_count'] = int(cnt)
                            if rec.get('installment_open_total') is None and amt is not None:
                                rec['installment_open_total'] = int(amt)
                        if 'revolv' in low or 'rev' in low:
                            if rec.get('revolving_open_count') is None and cnt is not None:
                                rec['revolving_open_count'] = int(cnt)
                            if rec.get('revolving_open_total') is None and amt is not None:
                                rec['revolving_open_total'] = int(amt)
                # if we found both, stop scanning
                if rec.get('installment_open_count') is not None and rec.get('revolving_open_count') is not None:
                    break
    except Exception:
        pass

    # Merge minimal file metadata and numeric fields into the record itself (no wide_row output)
    rec['filename'] = pdf_path.name
    rec['source'] = str(pdf_path)
    # ensure numeric fields exist even if not found
    rec['installment_open_count'] = rec.get('installment_open_count', '')
    rec['installment_open_total'] = rec.get('installment_open_total', '')
    rec['revolving_open_count'] = rec.get('revolving_open_count', '')
    rec['revolving_open_total'] = rec.get('revolving_open_total', '')

    # Delegate factor extraction to the main extractor (which uses column logic + normalization)
    try:
        rec = extract_record_level(full_text, doc=doc)
    except Exception:
        # fallback: keep rec populated from previous text parsing and try a local normalization
        try:
            rec['credit_factors'] = normalize_factors(factors)
        except Exception:
            rec['credit_factors'] = [{'label': f.get('factor'), 'canonical': map_line_to_canonical(f.get('factor')), 'count': None, 'total': None, 'color': f.get('color'), 'hex': f.get('hex','')} for f in factors]
    # also expose parsed public records from full_text if available
    try:
        pr_count, pr_note = parse_public_records(full_text)
        rec['public_records'] = pr_count
        rec['public_record_note'] = pr_note
    except Exception:
        rec['public_records'] = rec.get('public_records',0)
        rec['public_record_note'] = rec.get('public_record_note','')

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

    outp = OUT / f"{pdf_path.name}.json"
    with open(outp,'w') as fh:
        json.dump({'rec': rec}, fh, indent=2)
    print('Wrote', outp)
print('Done')
