#!/usr/bin/env python3
"""Rebuild `script_<stem>.json` from token-level span outputs in
`data/color_training/<stem>.p1.right.txt` using span-only color logic.
This avoids any phrase-based overrides and matches the intended span-first policy.
"""
import json, sys
from pathlib import Path
import colorsys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CT = ROOT / 'data' / 'color_training'
AV = ROOT / 'data' / 'agent_validation'
from PIL import Image
import fitz
PDF_DIR = ROOT / 'data' / 'pdf_analysis'
SPN_WHITE_THRESH = 740
# increase saturation threshold to avoid mapping low-saturation grays to colored categories
SPAN_SAT_MIN = 0.08
CANONICAL = {'green': (44,160,44), 'red': (204,0,0)}


def color_distance(a,b):
    return sum((int(x)-int(y))**2 for x,y in zip(a,b))


def rgb_to_hex(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def map_color_simple(rgb):
    if rgb is None:
        return 'neutral'
    r,g,b = [x/255.0 for x in rgb]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    hue = h*360
    if v < 0.12 and s < 0.25:
        return 'black'
    if s >= SPAN_SAT_MIN:
        if 60 <= hue <= 170:
            return 'green'
        if hue <= 30 or hue >= 330:
            return 'red'
        if 30 < hue < 60:
            return 'red'
        return 'neutral'
    # fallback to distance to canonical
    best=None; bestd=None
    for name, canon in CANONICAL.items():
        d = color_distance(tuple(int(x) for x in rgb), canon)
        if best is None or d < bestd:
            best, bestd = name, d
    if bestd is not None:
        if s < 0.12:
            # For low-saturation colors require a tighter distance to canonical to call it colored
            if bestd < 40000:
                return best
        else:
            if bestd < 70000:
                return best
    # allow looser match when saturation is slightly higher (>0.01) and reasonably close
    if s >= 0.01 and bestd is not None and bestd < 120000:
        return best
    # very light with green bias
    r2,g2,b2 = tuple(int(x*255) for x in (r,g,b))
    if v > 0.85 and (g2 - r2) > 8 and (g2 - b2) > 8:
        return 'green'
    return 'neutral'


def load_tokens(base):
    f = CT / f"{base}.p1.right.txt"
    if not f.exists():
        raise FileNotFoundError(f)
    toks = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # group by top coordinate into lines
    lines = []
    for t in toks:
        top = t.get('top')
        placed = False
        for ln in lines:
            if abs(ln['top'] - top) <= 6:
                ln['tokens'].append(t)
                placed=True; break
        if not placed:
            lines.append({'top': top, 'tokens':[t]})
    # sort tokens left-to-right and build phrases
    out = []
    for ln in sorted(lines, key=lambda x:x['top']):
        toks = sorted(ln['tokens'], key=lambda x: x.get('left',0))
        phrase = ' '.join((t.get('token') or '').strip() for t in toks).strip()
        rgbs = [t.get('rgb') for t in toks if isinstance(t.get('rgb'), list)]
        out.append({'phrase': phrase, 'rgbs': rgbs})
    return out


def _sample_phrase_bbox_from_pdf(stem, phrase, zoom=6, white_threshold=740):
    """Render the page for `stem` and sample the median of non-white pixels within the phrase bbox.
    Returns (hex, rgb) or (None, None).
    The function looks up the phrase by simple substring match in the rendered page's text lines.
    """
    pdf = PDF_DIR / f"{stem}.pdf"
    if not pdf.exists():
        return None, None
    try:
        doc = fitz.open(str(pdf))
        page = doc[0]
        td = page.get_text('dict')
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                t = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if phrase.strip().lower() in t.lower():
                    spans = ln.get('spans', [])
                    xs0 = [s.get('bbox')[0] for s in spans if s.get('bbox')]
                    ys0 = [s.get('bbox')[1] for s in spans if s.get('bbox')]
                    xs1 = [s.get('bbox')[2] for s in spans if s.get('bbox')]
                    ys1 = [s.get('bbox')[3] for s in spans if s.get('bbox')]
                    if not xs0:
                        return None, None
                    lx0,ly0,lx1,ly1 = min(xs0), min(ys0), max(xs1), max(ys1)
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                    except Exception:
                        return None, None
                    sx0 = max(0, int(lx0 * zoom) - 2); sy0 = max(0, int(ly0 * zoom) - 2)
                    sx1 = min(img.width, int(lx1 * zoom) + 2); sy1 = min(img.height, int(ly1 * zoom) + 2)
                    if sx1 <= sx0 or sy1 <= sy0:
                        return None, None
                    crop = img.crop((sx0, sy0, sx1, sy1))
                    arr = np.array(crop).reshape(-1,3)
                    sel = arr[arr.sum(axis=1) < white_threshold]
                    if sel.size == 0:
                        return None, None
                    med = tuple(map(int, np.median(sel, axis=0)))
                    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}", med
    except Exception:
        return None, None


def choose_color_from_spans(rgbs):
    if not rgbs:
        return None, None
    # prefer non-white candidates
    candidates = [tuple(int(x) for x in rgb) for rgb in rgbs if sum(rgb) < SPN_WHITE_THRESH]
    if not candidates:
        candidates = [tuple(int(x) for x in rgb) for rgb in rgbs]
    # prefer any that map to colored category
    colored = []
    for rgb in candidates:
        cat = map_color_simple(rgb)
        if cat in ('green','red'):
            colored.append(rgb)
            continue
        # allow pale tints close to canonical
        if min(color_distance(rgb, CANONICAL['green']), color_distance(rgb,CANONICAL['red'])) < 90000:
            colored.append(rgb)
    use = colored or candidates
    med = tuple(int(sum(c[i] for c in use)/len(use)) for i in range(3))
    cat = map_color_simple(med)
    # treat black as neutral for comparison purposes
    if cat == 'black':
        cat = 'neutral'
    # if median is very low-saturation and relatively light, prefer neutral (avoid mapping pale grays via distance)
    r,g,b = [x/255.0 for x in med]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    if s < 0.03 and v > 0.7:
        return None, 'neutral'
    return rgb_to_hex(med), cat


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: rebuild_script_from_spans.py <user_stem>')
        sys.exit(2)
    stem = sys.argv[1]
    lines = load_tokens(stem)
    out = []
    for ln in lines:
        # avoid labeling short numeric tokens as colored (they are usually bullets/labels)
        if ln['phrase'].strip().isdigit() and len(ln['phrase'].strip()) <= 2:
            hexv, cat = None, 'neutral'
        else:
            hexv, cat = choose_color_from_spans(ln['rgbs'])
            # If spans were all near-white or the result is neutral, try a rendered-phrase bbox sampling fallback
            all_white = ln['rgbs'] and all(sum(rgb) >= SPN_WHITE_THRESH for rgb in ln['rgbs'])
            if (cat == 'neutral' or hexv is None) and all_white:
                res = _sample_phrase_bbox_from_pdf(stem, ln['phrase'])
                if res:
                    pdf_hex, pdf_rgb = res
                    if pdf_hex and pdf_rgb:
                        # map rgb to simple category using existing mapper
                        cat_pdf = map_color_simple(pdf_rgb)
                        if cat_pdf == 'black':
                            cat_pdf = 'neutral'
                        hexv, cat = pdf_hex, cat_pdf
            # if span-derived color is colored but has very low saturation, prefer sampling bbox to avoid mislabeling pale grays
            if cat in ('green','red') and hexv:
                try:
                    med_rgb = tuple(int(hexv[i:i+2],16) for i in (1,3,5))
                    r,g,b = [x/255.0 for x in med_rgb]
                    import colorsys as _cs
                    h,s,v = _cs.rgb_to_hsv(r,g,b)
                except Exception:
                    s = 1.0
                if s < 0.12:
                    res = _sample_phrase_bbox_from_pdf(stem, ln['phrase'])
                    if res:
                        pdf_hex, pdf_rgb = res
                        if pdf_hex and pdf_rgb:
                            cat_pdf = map_color_simple(pdf_rgb)
                            if cat_pdf == 'black':
                                cat_pdf = 'neutral'
                            # if bbox sampling disagrees (neutral), prefer bbox result
                            if cat_pdf != cat:
                                hexv, cat = pdf_hex, cat_pdf
        out.append({'factor': ln['phrase'], 'color': cat, 'hex': hexv or ''})
    outf = AV / f'script_{stem}.json'
    outf.write_text(json.dumps(out, indent=2))
    print('Wrote', outf)
