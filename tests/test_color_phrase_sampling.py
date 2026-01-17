import pytest
from pathlib import Path
import json

pytest.importorskip('fitz')
from src.pymupdf_compat import fitz

from src.scripts.pdf_color_extraction import combined_sample_color_for_phrase, map_color_to_cat

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / 'data' / 'color_training'
PDF_DIR = ROOT / 'data' / 'pdf_analysis'


def test_phrase_sampling_matches_png_annotations():
    # For a small sample of phrase files, ensure combined_sample_color_for_phrase matches phrase annotations
    phrase_files = list(TRAIN_DIR.glob('*.p1.right.phrases.txt'))[:10]
    assert phrase_files, 'No phrase files found in data/color_training'
    for pf in phrase_files:
        # derive pdf filename from phrase file name
        name = pf.name.replace('.p1.right.phrases.txt','')
        # try in pdf_analysis dir
        pdf_path = PDF_DIR / (name)
        if not pdf_path.exists():
            # fallback: try stripping additional suffixes
            possible = [p for p in PDF_DIR.glob(name+'*')]
            if possible:
                pdf_path = possible[0]
        if not pdf_path.exists():
            pytest.skip(f'PDF not found for {name}')
        doc = fitz.open(str(pdf_path))
        with open(pf, 'r') as fh:
            lines = [json.loads(l) for l in fh]
        # pick target phrases with non-neutral colors
        # Skip tiny/ambiguous single-token numeric phrases (e.g., '1') which are noisy
        targets = [l for l in lines if l['cat'] in ('red','green','amber') and len(l['phrase'].strip())>1 and not l['phrase'].strip().isdigit()]
        if not targets:
            pytest.skip(f'No colored phrases in {pf}')
        for t in targets[:3]:
            phrase = t['phrase']
            expected = t['cat']
            res = combined_sample_color_for_phrase(doc, phrase, expected_color=None, page_limit=1)
            assert res is not None, f'Phrase not found: {phrase} in {name}'
            pidx, text, hexv, rgb, bbox, method = res
            cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
            assert cat == expected, f'Color mismatch for {phrase} in {name}: expected {expected} got {cat} (method={method})'
