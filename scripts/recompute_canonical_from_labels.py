#!/usr/bin/env python3
"""Recompute canonical color centers from human annotations (data/labels/*) and write suggestions to data/label_canonicals.json
Run after annotating and exporting crops via the annotation UI.
"""
from pathlib import Path
from PIL import Image
import json
ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'data' / 'labels'
OUT = ROOT / 'data' / 'label_canonicals.json'
res = {}
for cat_dir in LABELS.iterdir():
    if not cat_dir.is_dir():
        continue
    colors = []
    for p in cat_dir.glob('*.png'):
        im = Image.open(p).convert('RGB')
        data = list(im.getdata())
        if not data:
            continue
        mean = tuple(int(sum(c[i] for c in data) / len(data)) for i in range(3))
        colors.append(mean)
    if colors:
        import numpy as np
        arr = np.array(colors)
        med = tuple(map(int, arr.mean(axis=0)))
        res[cat_dir.name] = med
if res:
    OUT.write_text(json.dumps(res, indent=2))
    print('Wrote suggested canonicals to', OUT)
else:
    print('No labeled crops found under', LABELS)
