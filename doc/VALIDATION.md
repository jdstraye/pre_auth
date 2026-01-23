# PDF → Ground Truth Validation Workflow

This document explains how to produce an *annotated* ground-truth file with spans/bboxes and how to manually validate the results.

1. Activate the repo venv:
   - source .venv_pre_auth/bin/activate

2. Generate a text-only ground-truth JSON (no spans):
   - python scripts/pdf_to_ground_truth.py data/pdf_analysis/user_1314_credit_summary_*.pdf
   - Output: `data/extracted/<user>_ground_truth_unvalidated.json` (text-only, minimal fields)

3. Generate an annotated ground-truth JSON (spans + canonical keys):
   - python scripts/pdf_to_ground_truth.py data/pdf_analysis/user_1314_credit_summary_*.pdf --include-spans
   - Output: `*_ground_truth_unvalidated.json` and an enriched `*.with_spans.json` that contains `bbox`, `page`, `spans`, and `canonical_key` for each matched factor.

4. Review mapped results for human validation:
   - Run the auto-mapper review CSV (optional):
     - python scripts/auto_map_unvalidated.py --input data/extracted --out regression/quick50_candidates.csv --take 50
   - Create visual crops for manual verification:
     - python scripts/preview_gt_spans.py data/extracted/<gt-file>.with_spans.json --out tmp/crops
     - Inspect `tmp/crops/<gt-stem>/preview.csv` and PNG images. The CSV contains `factor,page,bbox,color,hex,spans` to quickly validate the mapping.

5. Manual edits and promotion:
   - After review, a validator may edit `*.with_spans.json` to correct `bbox`/`spans`/`canonical_key` entries and then save as `*_ground_truth.json` (validated) and commit.

Notes:
- The extractor's default output is minimal and production-friendly; rich annotation (spans/bboxes) is attached only when explicitly requested via `--include-spans`.
- `color` is the canonical color (one of `red|green|black|neutral`). `hex` contains the actual hex color (e.g., `#ff0000`) when measurable.
