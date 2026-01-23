"""Extract line-level spans (text, bbox, hex color, font size, bold) from PDFs using PyMuPDF.

Writes a companion JSON alongside the PDF: same-name + '.lines.json' in `data/pdf_analysis/`.
This is a lightweight, deterministic extractor intended to feed `auto_map_unvalidated.py` when precomputed
line-level data is missing.

Usage:
    PYTHONPATH=. python3 scripts/attach_spans_from_pdf.py data/pdf_analysis/user_123*.pdf

Output:
    data/pdf_analysis/user_123_credit_summary_...pdf.lines.json

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Allow running as `python scripts/attach_spans_from_pdf.py` without setting PYTHONPATH explicitly
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pymupdf_compat import fitz


def span_from_text_span(ts: Any) -> Dict[str, Any]:
    # ts is a dict-like returned by page.get_text('dict') structure
    return {
        'text': ts.get('text', ''),
        'bbox': [float(v) for v in ts.get('bbox', [])],
        'font_size': ts.get('size'),
        'font': ts.get('font'),
        'is_bold': (ts.get('flags', 0) & 2) != 0 if 'flags' in ts else False,
        'hex': None  # PyMuPDF doesn't expose color per span via get_text('dict') reliably
    }


def extract_lines(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    out: List[Dict[str, Any]] = []
    for pidx, page in enumerate(doc, start=0):
        # get text as dict which has blocks -> lines -> spans
        try:
            d = page.get_text('dict')
        except Exception:
            # older PyMuPDF fallback
            txt = page.get_text()
            out.append({'page': pidx, 'spans': [], 'line_text': txt, 'bbox': None})
            continue
        for block in d.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                spans = [span_from_text_span(s) for s in line.get('spans', [])]
                line_text = ''.join(s.get('text', '') for s in spans).strip()
                # compute bbox as union of span bboxes if present
                bboxes = [s['bbox'] for s in spans if s.get('bbox')]
                bbox = None
                if bboxes:
                    x0 = min(b[0] for b in bboxes)
                    y0 = min(b[1] for b in bboxes)
                    x1 = max(b[2] for b in bboxes)
                    y1 = max(b[3] for b in bboxes)
                    bbox = [x0, y0, x1, y1]
                out.append({'page': pidx, 'spans': spans, 'line_text': line_text, 'bbox': bbox})
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('pdfs', nargs='+', help='PDF file(s) to process')
    args = parser.parse_args(argv)
    for p in args.pdfs:
        pth = Path(p)
        if not pth.exists():
            print('missing:', p)
            continue
        lines = extract_lines(pth)
        outp = pth.with_suffix(pth.suffix + '.lines.json')
        outp.write_text(json.dumps({'pdf': str(pth), 'lines': lines}, indent=2), encoding='utf-8')
        print('wrote', outp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
