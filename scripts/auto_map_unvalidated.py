"""Auto-map unvalidated ground-truth JSONs to PDF line/spans.

Purpose
- Attach `page`, `bbox`, `spans` and deterministic `canonical_key` to factors found in
  `_ground_truth_unvalidated.json` by matching factor text against extractor lines.
- Emit a per-factor CSV for quick human review and a mapped JSON (written to `tmp/`).

Usage
    python scripts/auto_map_unvalidated.py --input data/extracted --out regression/quick50_candidates.csv --take 50

Notes
- Matching is conservative: exact (case-insensitive) first, then fuzzy (difflib) with an adjustable threshold.
- This tool does NOT promote files to validated; it produces a review CSV and mapped JSONs for manual approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import csv
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple

# Allow running as `python scripts/auto_map_unvalidated.py` without setting PYTHONPATH explicitly
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.pdf_color_extraction import extract_pdf_all_fields

OUT_DIR = pathlib.Path("tmp/auto_map")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    s2 = ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in s).strip()
    s2 = ' '.join(s2.split())
    return s2.replace(' ', '_')


def canonical_key_for(text: str) -> str:
    base = slugify(text)
    h = hashlib.sha1(base.encode('utf-8')).hexdigest()[:10]
    return f"{base}--{h}"


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a or '', b or '').ratio()


def find_best_line_match(factor_text: str, doc_lines: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """Return (line_obj, score, match_type).
    score in [0..1]. match_type is 'exact'|'substring'|'fuzzy'|'none'.
    """
    norm_f = factor_text.strip()
    # exact equality
    for ln in doc_lines:
        line_text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
        if line_text == norm_f:
            return ln, 1.0, 'exact'
    # substring (case-insensitive)
    for ln in doc_lines:
        line_text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
        if norm_f.lower() in line_text.lower() or line_text.lower() in norm_f.lower():
            return ln, 0.9, 'substring'
    # fuzzy
    best = None
    best_score = 0.0
    for ln in doc_lines:
        line_text = ''.join([s.get('text', '') for s in ln.get('spans', [])]).strip()
        score = fuzzy_ratio(norm_f.lower(), line_text.lower())
        if score > best_score:
            best_score = score
            best = ln
    if best_score >= 0.65:
        return best, best_score, 'fuzzy'
    return None, 0.0, 'none'


def load_doc_lines(pdf_path: str) -> List[Dict[str, Any]]:
    """Run extractor and return list of line dicts (with 'spans' and 'bbox').

    Note: request `include_spans=True` so returned line dicts include per-span color/hex/bbox metadata
    which the auto-mapper relies upon to attach span-level annotations to ground-truth entries.
    """
    # ask the extractor for spans explicitly so mapping has full context
    rec = extract_pdf_all_fields(str(pdf_path), include_spans=True)
    # extract_pdf_all_fields returns a dict; many helpers build `all_lines`/`all_spans`.
    # We rely on the extractor's internal `get_candidates_for_phrase` shape by re-running a light extraction.
    # The extractor currently doesn't expose `all_lines` directly via the public API, so we re-run lower-level flow
    # by opening the PDF and using get_candidates_for_phrase via the existing API is also an option.
    # For robustness, call extract_pdf_all_fields then (if available) try to read accompanying pdf-analysis file.
    # Fallback: return an empty list (caller will flag for manual review).
    # NOTE: extract_pdf_all_fields writes to disk in some callers; here we rely on its in-memory return.
    lines: List[Dict[str, Any]] = []
    # try to read an attached debug structure if present
    if isinstance(rec, dict):
        # Some extractor variants include `__line_debug__` or `poc_lines` keys; attempt to find plausible lists
        for k in ('all_lines_obj', 'lines', 'poc_lines', 'line_blocks'):
            if k in rec and isinstance(rec[k], list):
                return rec[k]
    # fallback: try to open a sibling pdf.json file produced by run_sample_extraction
    pdf_json = pathlib.Path(str(pdf_path)).with_suffix('.pdf.json')
    if pdf_json.exists():
        j = json.loads(pdf_json.read_text())
        # look for a list of lines with spans
        for k in ('lines', 'all_lines', 'line_items'):
            if k in j and isinstance(j[k], list):
                return j[k]
    # also accept a companion `.lines.json` produced by `attach_spans_from_pdf.py`
    companion = pathlib.Path(str(pdf_path)).with_suffix(pdf_path.suffix + '.lines.json')
    if companion.exists():
        j = json.loads(companion.read_text())
        if 'lines' in j and isinstance(j['lines'], list):
            return j['lines']
    return lines


def map_file(gt_path: str) -> Tuple[pathlib.Path, List[Dict[str, Any]]]:
    gt_path = pathlib.Path(gt_path)
    doc_pdf = None
    # try to infer pdf path from gt file
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    src = gt.get('source') or gt.get('filename')
    if src:
        # prefer absolute path when available
        pdf_path = pathlib.Path(src) if os.path.exists(src) else pathlib.Path('data/pdf_analysis') / pathlib.Path(src).name
        if not pdf_path.exists():
            # try matching by user id pattern
            candidates = list(pathlib.Path('data/pdf_analysis').glob(f"{gt_path.stem.split('_')[0]}*.pdf"))
            pdf_path = candidates[0] if candidates else None
    else:
        pdf_path = None
    doc_lines = []
    if pdf_path and pdf_path.exists():
        # use extractor to get line/spans
        try:
            doc_lines = load_doc_lines(pdf_path)
        except Exception:
            doc_lines = []
    out_rows: List[Dict[str, Any]] = []
    factors = gt.get('credit_factors', [])
    for f in factors:
        factor_text = f.get('factor') or f.get('text') or ''
        color_hex = f.get('hex') or f.get('color')
        # include top-level metadata from GT so review CSV has context (age, credit_score, address, filename)
        mapped = {
            'input_file': str(gt_path),
            'filename': gt.get('filename') or gt.get('source') or '',
            'credit_score': gt.get('credit_score'),
            'age': gt.get('age'),
            'address': gt.get('address'),
            'factor': factor_text,
            'color': color_hex,
            'canonical_key': '',
            'page': None,
            'bbox': None,
            'spans': None,
            'match_type': 'none',
            'match_score': 0.0,
            'notes': ''
        }
        if doc_lines:
            ln, score, mtype = find_best_line_match(factor_text, doc_lines)
            if ln is not None:
                mapped['page'] = ln.get('page') or ln.get('p') or 0
                mapped['bbox'] = ln.get('bbox')
                mapped['spans'] = ln.get('spans')
                mapped['match_type'] = mtype
                mapped['match_score'] = float(score)
                mapped['canonical_key'] = canonical_key_for(factor_text)
                # color agreement check (if spans present)
                hexv = None
                spans = ln.get('spans', [])
                if spans:
                    # prefer first non-empty span hex
                    for s in spans:
                        if s.get('hex'):
                            hexv = s.get('hex'); break
                mapped['color_match'] = (hexv == color_hex) if (hexv and color_hex) else None
            else:
                mapped['notes'] = 'no_line_match'
        else:
            mapped['notes'] = 'no_extracted_lines'
        out_rows.append(mapped)
    # write mapped JSON for review
    out_json = OUT_DIR / (gt_path.name.replace('.json', '.mapped.json'))
    out_json.write_text(json.dumps({'source_gt': str(gt_path), 'mapped': out_rows}, indent=2), encoding='utf-8')
    return out_json, out_rows


def discover_unvalidated(input_dir: str) -> List[pathlib.Path]:
    p = pathlib.Path(input_dir)
    return sorted(p.glob('*_ground_truth_unvalidated.json'))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/extracted', help='directory with _ground_truth_unvalidated.json files')
    ap.add_argument('--out', default='regression/quick50_candidates.csv', help='CSV output for review')
    ap.add_argument('--take', type=int, default=50, help='max files to process')
    ap.add_argument('--priority', nargs='*', help='space-separated user ids to prioritize (e.g., user_1314)')
    args = ap.parse_args(argv)

    files = discover_unvalidated(args.input)
    # prioritize
    if args.priority:
        pr = [p for u in args.priority for p in files if u in p.name]
        remaining = [p for p in files if p not in pr]
        files = pr + remaining
    files = files[: args.take]

    out_rows_all: List[Dict[str, Any]] = []
    for f in files:
        out_json, rows = map_file(str(f))
        for r in rows:
            r['mapped_json'] = str(out_json)
            out_rows_all.append(r)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['input_file','factor','color','canonical_key','page','bbox','match_type','match_score','color_match','mapped_json','notes'])
        writer.writeheader()
        for r in out_rows_all:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items() if k in writer.fieldnames})
    print(f"Wrote {len(out_rows_all)} factor rows to {outp}; mapped JSONs in {OUT_DIR}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
