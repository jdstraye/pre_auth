#!/usr/bin/env python3
"""Prepare a prioritized annotation queue from the latest `data/poc_qa_report.csv`.

Creates symlinks in `data/poc_imgs/` prefixed with `PRIOR_NN_` pointing to ambiguous QA crop images
for the specified record (defaults to 1314). Also writes `data/annotation_queue.json` with the list.
"""
from pathlib import Path
import csv
import argparse
import os

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / 'data' / 'poc_imgs'
QA_REPORT = ROOT / 'data' / 'poc_qa_report.csv'
QUEUE_FILE = ROOT / 'data' / 'annotation_queue.json'

parser = argparse.ArgumentParser()
parser.add_argument('--record', default='1314', help='record id to queue (default: 1314)')
parser.add_argument('--limit', type=int, default=40, help='max items to queue')
args = parser.parse_args()

if not QA_REPORT.exists():
    print('No QA report found at', QA_REPORT)
    raise SystemExit(1)

items = []
with open(QA_REPORT) as fh:
    r = csv.DictReader(fh)
    for row in r:
        fn = row.get('filename','')
        if args.record not in fn:
            continue
        # select ambiguous/mismatched rows
        expected = row.get('expected','')
        sampled = row.get('sampled_cat','')
        ok = row.get('ok','')
        # prioritize mismatched and neutral samples
        score = 0
        if ok in ('False','0',''):
            score += 10
        if sampled == 'neutral' or sampled == '' or sampled is None:
            score += 5
        # also include cases not found in rows
        if row.get('found_in_rows','') != 'True':
            score += 2
        if score == 0:
            continue
        sample_img = row.get('sample_img','')
        if not sample_img:
            continue
        p = Path(sample_img)
        if not p.exists():
            continue
        items.append({'phrase': row.get('phrase'), 'expected': expected, 'sampled': sampled, 'img': p, 'score': score})

# sort by score desc
items.sort(key=lambda x: -x['score'])
items = items[:args.limit]

# create prioritized symlinks
created = []
for i, it in enumerate(items, start=1):
    src = it['img']
    base = src.name
    dest = IMG_DIR / f"PRIOR_{i:02d}_{base}"
    try:
        if dest.exists():
            dest.unlink()
        os.symlink(os.path.relpath(src, IMG_DIR), dest)
        created.append(str(dest.name))
    except Exception:
        try:
            # fallback to copy
            from shutil import copyfile
            copyfile(src, dest)
            created.append(str(dest.name))
        except Exception:
            pass

# write queue file
import json
QUEUE_FILE.write_text(json.dumps({'record': args.record, 'items': [{'img': n, 'phrase':it['phrase'], 'expected':it['expected'], 'sampled':it['sampled']} for n, it in zip(created, items)]}, indent=2))
print('Created', len(created), 'queue items; open http://127.0.0.1:5000/ to annotate (or run scripts/annotation_ui.py)')
