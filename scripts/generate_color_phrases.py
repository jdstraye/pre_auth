"""Generate phrase-level color labels from span-level txt + right-half PNGs in data/color_training

For each *p1.right.txt and corresponding *p1.right.png, groups spans into phrases (by line top and adjacency),
computes median color of non-white pixels in the phrase bbox of the PNG, and writes a *.phrases.txt JSON-lines file.
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / 'data' / 'color_training'

WHITE_THRESHOLD = 740


def load_spans(txt_path):
    spans = []
    with open(txt_path, 'r') as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
                spans.append(obj)
            except Exception:
                pass
    return spans


def group_lines(spans, top_tol=6):
    # group by top coordinate with tolerance
    lines = []
    for s in sorted(spans, key=lambda x: (x['top'], x['left'])):
        placed=False
        for ln in lines:
            if abs(ln['top'] - s['top']) <= top_tol:
                ln['spans'].append(s)
                ln['top'] = int(round((ln['top'] * (len(ln['spans'])-1) + s['top']) / len(ln['spans'])))
                placed=True
                break
        if not placed:
            lines.append({'top': s['top'], 'spans': [s]})
    # sort spans in each line by left
    for ln in lines:
        ln['spans']=sorted(ln['spans'], key=lambda x: x['left'])
    return lines


def group_phrases_in_line(line, gap_tol=12):
    phrases=[]
    curr=[line['spans'][0]]
    for a,b in zip(line['spans'], line['spans'][1:]):
        gap = b['left'] - (a['left'] + a['width'])
        if gap <= gap_tol:
            curr.append(b)
        else:
            phrases.append(curr)
            curr=[b]
    phrases.append(curr)
    return phrases


def sample_bbox_png(png_path, bbox_px, white_threshold=WHITE_THRESHOLD):
    img=Image.open(png_path).convert('RGB')
    x0,y0,x1,y1=bbox_px
    w,h=img.size
    x0=max(0,int(x0)); y0=max(0,int(y0)); x1=min(w,int(x1)); y1=min(h,int(y1))
    if x1<=x0 or y1<=y0:
        return None
    crop=img.crop((x0,y0,x1,y1))
    arr=np.array(crop).reshape(-1,3)
    if arr.size==0:
        return None
    sel=arr[arr.sum(axis=1) < white_threshold]
    if sel.size==0:
        sel=arr
    med=tuple(map(int, np.median(sel, axis=0)))
    return med


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def map_color_to_cat_simple(rgb):
    # Use existing heuristics in poc script: rely on saturation/value heuristics
    try:
        r,g,b = [x/255.0 for x in rgb]
        mx=max(r,g,b); mn=min(r,g,b)
        if mx==0:
            return 'neutral'
        s=(mx-mn)/mx
        import colorsys
        h,_,v = colorsys.rgb_to_hsv(r,g,b)
        if v < 0.12:
            return 'black'
        if s < 0.02:
            return 'neutral'
        # green-ish
        if g > r and g > b and g > 0.4:
            return 'green'
        if r > g and r > b and r > 0.4:
            return 'red'
        # fallback to neutral
        return 'neutral'
    except Exception:
        return 'neutral'


def main():
    txts=list(TRAIN_DIR.glob('*.p1.right.txt'))
    for t in txts:
        png=t.with_suffix('.png')
        if not png.exists():
            print('missing png for', t)
            continue
        spans=load_spans(t)
        lines=group_lines(spans)
        out_lines=[]
        for ln in lines:
            phrases=group_phrases_in_line(ln)
            for phspans in phrases:
                text=' '.join([s.get('token','') for s in phspans]).strip()
                # build bbox in px (these are already pixel coords in the .txt excerpts)
                lefts=[s.get('left') for s in phspans]
                tops=[s.get('top') for s in phspans]
                rights=[s.get('left')+s.get('width') for s in phspans]
                bottoms=[s.get('top')+s.get('height') for s in phspans]
                bbox_px=(min(lefts), min(tops), max(rights), max(bottoms))
                rgb=sample_bbox_png(png, bbox_px)
                if rgb is None:
                    continue
                hexv=rgb_to_hex(rgb)
                cat=map_color_to_cat_simple(rgb)
                out_lines.append({'phrase': text, 'hex': hexv, 'rgb':rgb, 'cat':cat, 'bbox': bbox_px})
        out_path=t.with_suffix('.phrases.txt')
        with open(out_path,'w') as fh:
            for o in out_lines:
                fh.write(json.dumps(o) + '\n')
        print('wrote', out_path)

if __name__ == '__main__':
    main()
