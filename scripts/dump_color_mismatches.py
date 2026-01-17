#!/usr/bin/env python3
"""Dump mismatches between PNG-labeled token colors and PDF-extracted colors.
Writes JSONL lines with token, expected, predicted, pdf, and saves small token crops for inspection.
"""
import pathlib, json, os, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.scripts.pdf_color_extraction import combined_sample_color_for_phrase, map_color_to_cat, median_5x5, rgb_to_hex_tuple
from PIL import Image

TRAIN_DIR = pathlib.Path('data/color_training')
PDF_DIR = pathlib.Path('data/pdf_analysis')
OUT_DIR = pathlib.Path('data/color_training/mismatches')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(txt_path):
    labels = []
    with open(txt_path,'r') as f:
        for line in f:
            try:
                js = json.loads(line)
                labels.append(js)
            except Exception:
                continue
    return labels


if __name__ == '__main__':
    txts = sorted(TRAIN_DIR.glob('*.p1.right.txt'))[:200]  # sample subset for quick inspection
    out_lines = []
    for txt in txts:
        labels = load_labels(txt)
        pdf = PDF_DIR / (txt.stem.replace('.p1.right','') + '.pdf')
        for lab in labels:
            token = lab['token']
            expected = lab['cat']
            res = combined_sample_color_for_phrase(str(pdf) and __import__('fitz').open(str(pdf)), token, page_limit=1)
            if res is None:
                predicted = 'none'
                method = None
            else:
                pidx, line_text, hexv, rgb, bbox, method = res
                predicted = map_color_to_cat(rgb)
            if predicted != expected:
                # save small crop from png for inspection
                png = TRAIN_DIR / (txt.stem.replace('.p1.right','') + '.p1.right.png')
                try:
                    img = Image.open(png)
                    # try using label bbox if present
                    if 'left' in lab:
                        l = max(0, lab['left']-4); t = max(0, lab['top']-4); r = l + lab['width'] + 8; b = t + lab['height'] + 8
                        crop = img.crop((l,t,r,b))
                    else:
                        crop = img
                    crop_path = OUT_DIR / (txt.stem + f"_{token[:10]}.png")
                    crop.save(crop_path)
                except Exception:
                    crop_path = None
                out = {'txt': txt.name, 'pdf': pdf.name, 'token': token, 'expected': expected, 'predicted': predicted, 'method': method, 'crop': str(crop_path) if crop_path else None}
                out_lines.append(out)
    out_path = OUT_DIR / 'mismatches_sample.jsonl'
    with out_path.open('w') as f:
        for o in out_lines:
            f.write(json.dumps(o) + "\n")
    print('Wrote', out_path, 'mismatches count:', len(out_lines))
