"""Produce CSV/JSONL for human labeling from provisional candidate snapshots.

Example:
  python scripts/prepare_labeling_csv.py --input data/regression_pool --out labeling/user_validate_user1314.csv --pdf-ids user_1314

Output columns (CSV): pdf_id,page,candidate_id,line_text,bbox,provisional_label,notes
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List


def load_provisionals(path: Path) -> List[dict]:
    out = []
    for p in sorted(path.glob("*_candidates_provisional.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        out.append(j)
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/regression_pool"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pdf-ids", type=str, help="comma-separated stems to include")
    args = p.parse_args(argv)

    provisionals = load_provisionals(args.input)
    rows = []
    for j in provisionals:
        if args.pdf_ids:
            ids = [x.strip() for x in args.pdf_ids.split(",")]
            if j.get("pdf_id") not in ids:
                continue
        for c in j.get("candidates", []):
            rows.append({
                "pdf_id": j.get("pdf_id"),
                "page": c.get("page"),
                "candidate_id": c.get("candidate_id"),
                "line_text": (c.get("line_text") or "").strip(),
                "bbox": json.dumps(c.get("bbox")),
                "provisional_label": "",
                "notes": "",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["pdf_id", "page", "candidate_id", "line_text", "bbox", "provisional_label", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
