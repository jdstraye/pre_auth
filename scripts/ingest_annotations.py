#!/usr/bin/env python3
"""Ingest saved annotations (data/annotations/*.json) and append entries to data/annotation_labels.csv
Each row: image, x0,y0,x1,y1, category, mean_r,mean_g,mean_b, orig_image
Also produces cropped examples under data/labels/<category>/ for quick training use.
"""
from pathlib import Path
import json
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / 'data' / 'annotations'
IMG_DIR = ROOT / 'data' / 'poc_imgs'
OUT_CSV = ROOT / 'data' / 'annotation_labels.csv'
OUT_LABELS = ROOT / 'data' / 'labels'
OUT_LABELS.mkdir(parents=True, exist_ok=True)
rows = []
for p in ANN_DIR.glob('*.json'):
    j = json.loads(p.read_text())
    img_name = j.get('image')
    im_p = IMG_DIR / img_name
    if not im_p.exists():
        continue
    im = Image.open(im_p).convert('RGB')
    for i, r in enumerate(j.get('rects', [])):
        x = int(r['x']); y = int(r['y']); w = int(r['w']); h = int(r['h'])
        if w <= 1 or h <= 1:
            continue
        cat = r.get('cat','neutral')
        crop = im.crop((x, y, x + w, y + h))
        data = list(crop.getdata())
        if not data:
            continue
        mean = tuple(int(sum(p[i] for p in data) / len(data)) for i in range(3))
        label_dir = OUT_LABELS / cat
        label_dir.mkdir(parents=True, exist_ok=True)
        fn = label_dir / f"{p.stem}_{i}.png"
        crop.save(fn)
        rows.append((img_name, x, y, x+w, y+h, cat, mean[0], mean[1], mean[2], str(fn)))
# append to CSV
if rows:
    import csv
    header = ['image','x0','y0','x1','y1','category','mean_r','mean_g','mean_b','crop_path']
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, 'a', newline='') as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f'Ingested {len(rows)} annotated regions; crops saved under {OUT_LABELS}')
else:
    print('No annotations to ingest.')
