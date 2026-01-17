#!/usr/bin/env python3
"""Generate color training dataset.
For each PDF in data/pdf_analysis, render the first page, crop the right half, save as PNG,
run Tesseract OCR to get token positions, sample median color around token centers from the PNG,
and write a .txt file with token,hex,r,g,b,category,x,y,width,height.
"""
import pathlib
import sys
# ensure repo root on path so we can import scripts.* modules when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fitz
from PIL import Image
import pytesseract
import numpy as np
import json
from src.scripts.pdf_color_extraction import median_5x5, rgb_to_hex_tuple, map_color_to_cat

PDF_DIR = pathlib.Path('data/pdf_analysis')
OUT_DIR = pathlib.Path('data/color_training')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_token_color_from_image(img, left, top, width, height):
    """Sample median 5x5 color around token center"""
    cx = int(left + width // 2)
    cy = int(top + height // 2)
    med = median_5x5(img, cx, cy)
    if med is None:
        arr = np.array(img)
        med = tuple(map(int, np.median(arr.reshape(-1,3), axis=0)))
    return med


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    # render at scale 2 for better OCR
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    w, h = img.size
    # crop right half
    midx = w // 2
    right = img.crop((midx, 0, w, h))
    out_png = OUT_DIR / (pdf_path.stem + '.p1.right.png')
    right.save(out_png)
    data = pytesseract.image_to_data(right, output_type=pytesseract.Output.DICT)
    tokens = []
    for i, txt in enumerate(data['text']):
        if not txt or txt.strip() == '':
            continue
        left = int(data['left'][i])
        top = int(data['top'][i])
        width = int(data['width'][i])
        height = int(data['height'][i])
        med = sample_token_color_from_image(right, left, top, width, height)
        hexv = rgb_to_hex_tuple(med)
        cat = map_color_to_cat(med)
        tokens.append({'token': txt, 'hex': hexv, 'rgb': med, 'cat': cat, 'left': left, 'top': top, 'width': width, 'height': height})
    out_txt = OUT_DIR / (pdf_path.stem + '.p1.right.txt')
    with out_txt.open('w') as f:
        for t in tokens:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {out_png} ({len(tokens)} tokens) and {out_txt}")


if __name__ == '__main__':
    pdfs = sorted(PDF_DIR.glob('*.pdf'))
    if not pdfs:
        print('No PDFs found in', PDF_DIR)
    for p in pdfs:
        try:
            process_pdf(p)
        except Exception as e:
            print('ERROR processing', p, e)
