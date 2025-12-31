#!/usr/bin/env python3
"""Compute robust canonical colors from labeled crops by ignoring near-white background.
Writes output to data/label_canonicals_filtered.json and prints suggestions.
"""
from pathlib import Path
from PIL import Image
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'data' / 'labels'
OUT = ROOT / 'data' / 'label_canonicals_filtered.json'
res = {}
for cat_dir in LABELS.iterdir():
    if not cat_dir.is_dir():
        continue
    colors = []
    for p in cat_dir.glob('*.png'):
        im = Image.open(p).convert('RGB')
        arr = np.array(im).reshape(-1,3)
        # select non-white-ish pixels
        sums = arr.sum(axis=1)
        mask = sums < 720  # exclude near-white backgrounds
        sel = arr[mask]
        if sel.size == 0:
            # fallback to entire crop
            sel = arr
        # compute median color of selected pixels
        med = tuple(map(int, np.median(sel, axis=0)))
        # require some saturation > 0.01 to consider useful
        r,g,b = [x/255.0 for x in med]
        import colorsys
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        if s < 0.01:
            # if too unsaturated, skip this crop
            continue
        colors.append(med)
    if colors:
        arr = np.array(colors)
        med = tuple(map(int, arr.mean(axis=0)))
        res[cat_dir.name] = med
if res:
    import json
    OUT.write_text(json.dumps(res, indent=2))
    print('Wrote suggested filtered canonicals to', OUT)
    print(res)
else:
    print('No usable labeled crops found for filtered canonical computation')
