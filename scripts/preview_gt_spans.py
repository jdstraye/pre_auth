"""Generate crops and span previews for GT JSON entries for quick visual verification.

This improved script:
- Includes top-level fields (monthly_payments, credit_freeze, fraud_alert, deceased, credit alerts, categories)
- Attempts to locate each phrase/entry in the source PDF and extract page, bbox, and spans
- Saves a crop image when a bbox is found
- Writes a CSV with canonical color (one of red/green/black/neutral) in the `hex` column for backward compatibility

Usage:
  python scripts/preview_gt_spans.py data/extracted/user_1131_credit_summary_2025-09-01_132805_ground_truth_unvalidated.json --out tmp/crops
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import sys

# ensure package root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pymupdf_compat import fitz

# Use helpers from the PDF color extraction module
try:
    from src.scripts.pdf_color_extraction import (
        combined_sample_color_for_phrase,
        color_first_search_for_phrase,
        span_color_hex,
        map_color_to_cat,
        find_credit_factors_region,
    )
except Exception:
    # Fail gracefully - script will still include top-level keys but won't locate spans
    combined_sample_color_for_phrase = None
    color_first_search_for_phrase = None
    span_color_hex = None
    map_color_to_cat = None
    find_credit_factors_region = None


def crop_page_bbox(page_obj, bbox, out_path: Path, zoom: int = 2):
    """Crop the page using bbox and save as PNG."""
    if bbox is None:
        return None
    rect = fitz.Rect(bbox)
    pix = page_obj.get_pixmap(clip=rect, matrix=fitz.Matrix(zoom, zoom))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return str(out_path)


def find_line_and_spans(doc, phrase, page_limit=3):
    """Locate a phrase in the document and return (page_idx, line_text, bbox, spans, rgb).

    Uses the aggressive HAMMER search first, then falls back to color-first search, and finally a
    naive textual match if neither is available.
    Returns None if nothing found.
    """
    # Try combined_sample_color_for_phrase (aggressive) then fallback
    res = None
    if combined_sample_color_for_phrase is not None:
        try:
            res = combined_sample_color_for_phrase(doc, phrase, page_limit=page_limit)
            # combined returns a tuple (page, line_text, hexv, rgb, bbox, tag)
            if res:
                pidx, line_text, hexv, rgb, bbox, *_ = res
            else:
                res = None
        except Exception:
            res = None
    if not res and color_first_search_for_phrase is not None:
        try:
            res2 = color_first_search_for_phrase(doc, phrase, page_limit=page_limit)
            if res2:
                # color_first_search_for_phrase returns (page, text, hex, rgb, bbox, pix_bbox, uncertain)
                pidx, line_text, hexv, rgb, bbox, *_ = res2
                res = (pidx, line_text, hexv, rgb, bbox)
        except Exception:
            res = None
    # As a last fallback, do a naive textual search of lines (no color)
    if not res:
        for pidx in range(min(page_limit, len(doc))):
            try:
                td = doc.load_page(pidx).get_text('dict')
                for b in td.get('blocks', []):
                    for ln in b.get('lines', []):
                        line_text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
                        if not line_text:
                            continue
                        if phrase.lower() in line_text.lower():
                            # pick the full line bbox and spans
                            hexv, rgb = span_color_hex(ln.get('spans', [])) if span_color_hex is not None else (None, None)
                            bbox = ln.get('bbox')
                            return pidx, line_text, bbox, ln.get('spans', []), rgb
            except Exception:
                continue
        return None
    # If we have a res tuple from earlier, extract spans by matching the returned line_text on that page
    pidx = res[0]
    line_text = res[1]
    hexv, rgb = res[2], res[3]
    bbox = res[4] if len(res) > 4 else None
    spans = []
    try:
        td = doc.load_page(pidx).get_text('dict')
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                ltxt = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
                if ltxt and ltxt.strip() == line_text.strip():
                    spans = ln.get('spans', [])
                    # prefer using the precise line bbox if bbox is none
                    if not bbox:
                        bbox = ln.get('bbox')
                    break
            if spans:
                break
    except Exception:
        spans = []
    return pidx, line_text, bbox, spans, rgb


def canonical_color_from_rgb(rgb, fallback=None):
    """Return canonical color name ('red','green','black','neutral') from rgb using map_color_to_cat when available."""
    if rgb is not None and map_color_to_cat is not None:
        return map_color_to_cat(rgb)
    if fallback:
        # fallback might be a provided color string
        return fallback if fallback in {'red', 'green', 'black', 'neutral'} else 'neutral'
    return 'neutral'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('gt')
    ap.add_argument('--out', default='tmp/crops')
    args = ap.parse_args(argv)
    gt_path = Path(args.gt)
    out_dir = Path(args.out) / gt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    j = json.loads(gt_path.read_text(encoding='utf-8'))
    pdf_path = Path(j.get('source') or j.get('filename'))
    if not pdf_path.exists():
        candidates = list(Path('data/pdf_analysis').glob(f"{gt_path.stem.split('_')[0]}*.pdf"))
        if candidates:
            pdf_path = candidates[0]
    doc = fitz.open(str(pdf_path))

    csv_lines = []
    idx = 0

    # Helper to append a CSV row
    def append_row(kind, key, value=None, page=None, bbox=None, image=None, color=None, hexv=None, spans=None):
        nonlocal idx
        csv_lines.append({
            'idx': idx,
            'kind': kind,
            'key': key,
            'value': value,
            'page': page,
            'bbox': bbox,
            'image': image,
            'color': color,
            'hex': hexv,
            'spans': json.dumps(spans, ensure_ascii=False) if spans else ''
        })
        idx += 1

    # Include top-level fields the user asked for even if we can't find bbox/spans
    top_keys = [
        'monthly_payments', 'credit_freeze', 'fraud_alert', 'deceased', 'credit_score',
        'credit_card_open_totals','revolving_open_total','installment_open_total','real_estate_open_total','line_of_credit_accounts_open_total',
        'public_records_details','late_pays_2yr','late_pays_gt2yr','inquiries_last_6_months','address','age','categories','credit_alerts'
    ]
    for k in top_keys:
        if k in j:
            val = j.get(k)
            p = None; b = None; img = None; spans = None; rgb = None
            # Special cases
            if k == 'monthly_payments' and isinstance(val, (int, float)) and val > 0:
                search_str = f"${int(val):,}"
                found = find_line_and_spans(doc, search_str)
                if not found:
                    found = find_line_and_spans(doc, str(int(val)))
                if found:
                    p, _, b, spans, rgb = found
                    if b is not None:
                        img = crop_page_bbox(doc[p], b, out_dir / f"{k}.png")
            elif k == 'credit_card_open_totals' and isinstance(j.get('credit_card_open_totals'), dict):
                # prefer captured spans in the GT
                cc = j.get('credit_card_open_totals')
                if cc.get('page') is not None:
                    p = cc.get('page'); b = cc.get('bbox'); spans = cc.get('spans'); rgb = None
                    img = crop_page_bbox(doc[p], b, out_dir / f"{k}.png") if b is not None else None
                else:
                    found = find_line_and_spans(doc, 'credit card open totals')
                    if found:
                        p, _, b, spans, rgb = found
                        if b is not None:
                            img = crop_page_bbox(doc[p], b, out_dir / f"{k}.png")
            elif k in ('public_records_details','credit_score') and isinstance(val, dict):
                # these may be dicts with detail/text; try matching by detail or number
                if 'detail' in val:
                    found = find_line_and_spans(doc, val.get('detail'))
                elif k == 'credit_score' and isinstance(val, (int, float)):
                    found = find_line_and_spans(doc, str(int(val)))
                else:
                    found = find_line_and_spans(doc, k.replace('_',' '))
                if found:
                    p, _, b, spans, rgb = found
                    if b is not None:
                        img = crop_page_bbox(doc[p], b, out_dir / f"{k}.png")
            else:
                found = find_line_and_spans(doc, k.replace('_', ' '))
                if found:
                    p, _, b, spans, rgb = found
                    if b is not None:
                        img = crop_page_bbox(doc[p], b, out_dir / f"{k}.png")
            color = canonical_color_from_rgb(rgb)
            append_row('top', k, value=val, page=p, bbox=b, image=img, color=color, hexv=color, spans=spans)

    # Attempt to locate 'credit alerts' and 'categories' sections (if present in text)
    for anchor in ['credit alerts', 'categories']:
        found = find_line_and_spans(doc, anchor)
        if found:
            p, line_text, b, spans, rgb = found
            img = crop_page_bbox(doc[p], b, out_dir / f"{anchor.replace(' ', '_')}.png") if b is not None else None
            color = canonical_color_from_rgb(rgb)
            append_row('section', anchor, value=line_text, page=p, bbox=b, image=img, color=color, hexv=color, spans=spans)

    # Process credit_factors list from GT
    for i, cf in enumerate(j.get('credit_factors', [])):
        factor = cf.get('factor') if isinstance(cf, dict) else str(cf)
        fallback_color = cf.get('color') if isinstance(cf, dict) else None
        found = find_line_and_spans(doc, factor)
        page = None; bbox = None; img = None; spans = None; rgb = None
        if found:
            page, line_text, bbox, spans, rgb = found
            if bbox is not None:
                img = crop_page_bbox(doc[page], bbox, out_dir / f"factor_{i}.png")
        color = canonical_color_from_rgb(rgb, fallback=fallback_color)
        append_row('factor', factor, value=None, page=page, bbox=bbox, image=img, color=color, hexv=color, spans=spans)

    # Fallback: include any remaining interesting lines in the 'Credit Factors' area if no explicit credit_factors provided
    if not j.get('credit_factors'):
        # look for the 'Credit Factors' block and capture lines under it
        for p in range(min(3, len(doc))):
            td = doc.load_page(p).get_text('dict')
            for b in td.get('blocks', []):
                for ln in b.get('lines', []):
                    line_text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
                    if line_text and 'credit factors' in line_text.lower():
                        # capture next few lines
                        # naive approach: capture the next 5 lines within the same block
                        ln_idx = b.get('lines', []).index(ln)
                        for jln in b.get('lines', [])[ln_idx+1:ln_idx+6]:
                            lt = ''.join([s.get('text', '') for s in jln.get('spans', [])]).strip()
                            if lt:
                                bx = jln.get('bbox')
                                sp = jln.get('spans', [])
                                img = crop_page_bbox(doc[p], bx, out_dir / f"credit_factor_fallback_{idx}.png") if bx is not None else None
                                c = None
                                try:
                                    hexv, rgb = span_color_hex(sp)
                                    c = canonical_color_from_rgb(rgb)
                                except Exception:
                                    c = 'neutral'
                                append_row('factor_fallback', lt, value=None, page=p, bbox=bx, image=img, color=c, hexv=c, spans=sp)
                        break

    # Write CSV with canonical color in 'hex' (backwards compat) and 'color' column
    import csv
    csvp = out_dir / 'preview.csv'
    fieldnames = ['idx', 'kind', 'key', 'value', 'page', 'bbox', 'image', 'color', 'hex', 'spans']
    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_lines:
            # ensure bbox is JSON serializable
            if hasattr(row['bbox'], '__iter__') and not isinstance(row['bbox'], str):
                row['bbox'] = json.dumps(row['bbox'])
            w.writerow(row)
    print('Wrote', len(csv_lines), 'preview rows and crops to', out_dir)


if __name__ == '__main__':
    main()
