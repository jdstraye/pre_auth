# Canonical PDF color extraction implementation
# (copied from src/scripts/pdf_color_extraction.py for new structure)

import argparse
import logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def cli_extract():
    parser = argparse.ArgumentParser(description="Extract all fields from a credit summary PDF.")
    parser.add_argument('--user_id', type=str, help='User ID (e.g., 705)')
    parser.add_argument('--pdf_dir', type=str, default='data/pdf_analysis', help='Directory containing PDFs')
    parser.add_argument('--output_dir', type=str, default='data/extracted', help='Directory to save extracted JSON')
    args = parser.parse_args()

    # Find PDF for user_id
    import glob
    pattern = f"{args.pdf_dir}/user_{args.user_id}_credit_summary_*.pdf"
    files = glob.glob(pattern)
    if not files:
        print(f"No PDF found for user {args.user_id}")
        return
    pdf_path = files[0]
    rec = extract_pdf_all_fields(pdf_path)
    out_path = f"{args.output_dir}/user_{args.user_id}_credit_summary_ground_truth_unvalidated.json"
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"Extracted fields saved to {out_path}")
if __name__ == "__main__":
    cli_extract()

try:
    from src.extraction.pymupdf_compat import fitz
except Exception:
    fitz = None  # Lazy import: tests that don't need PDF parsing can still import this module

# ...rest of canonical code (functions, logic, etc.)...
