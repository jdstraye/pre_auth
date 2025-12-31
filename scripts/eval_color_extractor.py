#!/usr/bin/env python3
"""Evaluate PDF-based color extraction against color_training labels.
For each training sample, try to locate the same token in the PDF (using combined_sample_color_for_phrase or span heuristics)
and compare sampled color/category to the ground-truth from the PNG.
Print per-file and aggregate accuracy.
"""
import pathlib, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.poc_extract_credit_factors import combined_sample_color_for_phrase, map_color_to_cat
import fitz

TRAIN_DIR = pathlib.Path('data/color_training')
PDF_DIR = pathlib.Path('data/pdf_analysis')


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


def find_in_pdf(pdf_path, token):
    doc = fitz.open(pdf_path)
    res = combined_sample_color_for_phrase(doc, token, page_limit=1)
    return res


if __name__ == '__main__':
    txts = sorted(TRAIN_DIR.glob('*.p1.right.txt'))
    total=0; correct_cat=0; span_confident=0; found=0
    perfile=[]
    for txt in txts:
        pdf = PDF_DIR / (txt.stem.replace('.p1.right','') + '.pdf')
        labels = load_labels(txt)
        f_total=0; f_correct=0; f_found=0
        for lab in labels:
            token = lab['token']
            expected_cat = lab['cat']
            res = find_in_pdf(pdf, token)
            f_total += 1
            total +=1
            if res is None:
                continue
            pidx, line_text, hexv, rgb, bbox, method = res
            found += 1; f_found +=1
            cat = map_color_to_cat(rgb)
            if cat == expected_cat:
                correct_cat += 1; f_correct +=1
        perfile.append((txt.name, f_found, f_correct, f_total))
    print('Per-file summary (file, found, correct, total):')
    for p in perfile:
        print(p)
    print('\nAggregate: total_tokens=', total, 'found=', found, 'correct=', correct_cat, 'acc=', float(correct_cat)/(found+1e-9))
