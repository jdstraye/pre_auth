"""Create a regression pool of candidate lines from PDFs for human validation.

Usage examples:
  # extract candidates for specific users (provisional labels from POC)
  python scripts/regen_regression_set.py --pdf-ids user_1314,user_582 --out-dir data/regression_pool --seed-from-poc --provisional

  # build a quick stratified subset for PR smoke-tests
  python scripts/regen_regression_set.py --quick-subset 50 --out-dir data/regression_pool

Outputs (by default):
  data/regression_pool/manifest.json  # index of generated artifacts
  data/regression_pool/<pdf_id>_candidates_provisional.json

This is a lightweight, review-first tool — it does NOT mark files as validated.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.scripts.pdf_color_extraction import extract_pdf_all_fields, extract_credit_factors_from_doc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REG_DIR = DATA_DIR / "regression_pool"
REG_DIR.mkdir(exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def candidate_snapshot_for_pdf(pdf_path: Path, seed_from_poc: bool = False) -> Dict[str, Any]:
    """Return a serializable snapshot of candidates for a PDF.

    Uses canonical extractor (`extract_pdf_all_fields`) to collect candidate-level data.
    If seed_from_poc is True, include the legacy POC output as `poc_candidates` for human review.
    """
    rec = extract_pdf_all_fields(str(pdf_path))
    candidates = rec.get("candidates", []) or rec.get("credit_factors", []) or []

    out = {
        "pdf_path": str(pdf_path),
        "pdf_id": pdf_path.stem,
        "candidates": [],
        "meta": {"generator": "regen_regression_set.py"},
    }

    for i, c in enumerate(candidates):
        # keep only compact, review-friendly fields
        out["candidates"].append({
            "candidate_id": f"c{i}",
            "page": c.get("page"),
            "line_text": c.get("line_text") or c.get("text"),
            "bbox": c.get("bbox"),
            "spans": c.get("spans", []),
            "color": c.get("color"),
            "canonical_key": c.get("canonical_key"),
        })

    if seed_from_poc:
        try:
            poc = extract_credit_factors_from_doc(str(pdf_path))
        except Exception:
            poc = None
        out["poc_candidates"] = poc

    return out


def list_candidate_pdfs() -> List[Path]:
    # prefer `data/pdf_analysis/*.pdf`, fallback to data/extracted presence
    pdfs = sorted((DATA_DIR / "pdf_analysis").glob("*.pdf"))
    if pdfs:
        return pdfs
    # fallback: try to infer from extracted jsons
    ex = sorted((DATA_DIR / "extracted").glob("*_credit_summary*.json"))
    ids = [p for p in ex if "ground_truth" not in p.name]
    return ids


def build_manifest(items: List[Path]) -> Dict[str, Any]:
    return {"count": len(items), "samples": [p.name for p in items]}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-ids", type=str, help="comma-separated pdf ids (filename stems)")
    p.add_argument("--out-dir", type=Path, default=REG_DIR)
    p.add_argument("--seed-from-poc", action="store_true")
    p.add_argument("--provisional", action="store_true", help="mark outputs as provisional (do not modify GT dirs)")
    p.add_argument("--quick-subset", type=int, help="build a small stratified subset for PR smoke tests")
    p.add_argument("--sample", type=int, help="random sample size (from available pdfs)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    available = list_candidate_pdfs()
    if args.pdf_ids:
        wanted = [ROOT / "data" / "pdf_analysis" / (x.strip() + ".pdf") for x in args.pdf_ids.split(",")]
    elif args.quick_subset:
        # simple deterministic sample: pick first N distinct templates (by filename hash)
        wanted = available[: args.quick_subset]
    elif args.sample:
        wanted = available[: args.sample]
    else:
        wanted = available

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_items = []

    for pdf in wanted:
        try:
            snap = candidate_snapshot_for_pdf(pdf, seed_from_poc=args.seed_from_poc)
        except Exception as exc:
            print(f"ERROR extracting {pdf}: {exc}")
            continue
        name = pdf.stem + ("_candidates_provisional.json" if args.provisional else "_candidates.json")
        dest = out_dir / name
        write_json(dest, snap)
        manifest_items.append(dest)
        if args.debug:
            print(f"wrote {dest} (candidates={len(snap['candidates'])})")

    manifest = build_manifest(manifest_items)
    write_json(out_dir / "manifest.json", manifest)
    print(f"wrote {len(manifest_items)} candidate snapshots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
