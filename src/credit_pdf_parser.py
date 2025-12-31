#!/usr/bin/env python3
"""
Credit Report → JSON Extractor
==============================

A **compact but heavily documented** pipeline that:

1. Uses **PP-StructureV2** (PaddleOCR) to detect table regions in a scanned PDF.
2. Crops each detected table with **pdfplumber**.
3. Parses the cropped image with **pdfplumber** + **regex** to extract rows.
4. Builds a **deterministic JSON structure** that mirrors the original report.

Why this combo?
- PP-StructureV2 is state-of-the-art for table detection on scanned docs.
- pdfplumber gives pixel-perfect text + line coordinates for post-processing.
- Pure regex avoids fragile OCR layout assumptions.

Author:  Your overly-explanatory software engineer
Date:    2025-10-31
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pdfplumber
from paddleocr import PPStructure, draw_structure_result
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# --------------------------------------------------------------------------- #
# 1. CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------------- #
# Adjust paths if needed
POPPLER_PATH = None  # Set if poppler isn't in PATH
DPI = 300            # Higher DPI → better OCR/table detection

# Table engine (PP-StructureV2)
TABLE_ENGINE = PPStructure(
    table=True,
    ocr=True,
    show_log=False,
    recovery=True,      # Helps with rotated/skewed tables
    use_pdf2docx_api=True,
)

# Output JSON schema version
JSON_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# 2. HELPER: Convert PDF page → PIL Image (for PP-Structure)
# --------------------------------------------------------------------------- #
def pdf_page_to_image(pdf_path: Path, page_num: int) -> Image.Image:
    """
    Convert a single PDF page to PIL Image using pdf2image.
    Page numbers are 1-based for pdf2image.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=DPI,
        first_page=page_num,
        last_page=page_num,
        poppler_path=POPPLER_PATH,
    )
    if not images:
        raise RuntimeError(f"Failed to convert page {page_num}")
    return images[0]

# --------------------------------------------------------------------------- #
# 3. TABLE DETECTION WITH PP-STRUCTUREV2
# --------------------------------------------------------------------------- #
def detect_tables_in_page(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Returns list of detected tables with bounding boxes in **pixel** coordinates.
    Each dict contains:
        - bbox: [x0, y0, x1, y1] (top-left, bottom-right)
        - type: "table"
        - res: raw PPStructure result (html, cells, etc.)
    """
    result = TABLE_ENGINE(image)
    tables = [item for item in result if item["type"].lower() == "table"]
    return tables

# --------------------------------------------------------------------------- #
# 4. CROP TABLE IMAGE USING BBOX
# --------------------------------------------------------------------------- #
def crop_table_image(image: Image.Image, bbox: List[float], margin: int = 10) -> Image.Image:
    """
    Crop the table from the full page image.
    Adds a small margin to avoid cutting off borders.
    """
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(image.width, x1 + margin)
    y1 = min(image.height, y1 + margin)
    return image.crop((x0, y0, x1, y1))

# --------------------------------------------------------------------------- #
# 5. EXTRACT TEXT GRID FROM CROPPED TABLE USING PDFPLUMBER
# --------------------------------------------------------------------------- #
def extract_table_grid(pdf: pdfplumber.PDF, page_num: int, bbox: List[float]) -> List[List[str]]:
    """
    Use pdfplumber to extract text in a grid layout **within** the detected bbox.
    Returns a list of rows → list of cleaned cell strings.
    """
    page = pdf.pages[page_num]
    # Convert bbox from image pixels → PDF points (1 inch = 72 points)
    # We need to scale: image is at DPI, PDF native is 72 dpi
    scale = 72 / DPI
    pdf_bbox = [coord * scale for coord in bbox]

    cropped = page.crop(pdf_bbox, relative=False)
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
    }
    try:
        table = cropped.extract_table(table_settings=table_settings)
    except Exception as e:
        print(f"Warning: pdfplumber table extraction failed: {e}", file=sys.stderr)
        table = None

    if table:
        # Clean and return
        return [[cell.strip() if cell else "" for cell in row] for row in table]

    # Fallback: extract words and group into pseudo-rows by y-coordinate
    words = cropped.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return []

    # Group words by approximate row (using median y)
    rows: Dict[float, List[Tuple[float, str]]] = {}
    for w in words:
        y_mid = (float(w["top"]) + float(w["bottom"])) / 2
        row_key = round(y_mid / 5) * 5  # bin every 5 points
        rows.setdefault(row_key, []).append((float(w["x0"]), w["text"]))

    grid: List[List[str]] = []
    for row_key in sorted(rows.keys()):
        row_words = rows[row_key]
        row_words.sort(key=lambda x: x[0])  # left → right
        grid.append([text for _, text in row_words])

    return [[cell.strip() for cell in row] for row in grid]

# --------------------------------------------------------------------------- #
# 6. CLEANING & PARSING UTILITIES
# --------------------------------------------------------------------------- #
def clean_currency(s: str) -> Optional[float]:
    """'$10,719' → 10719.0; returns None if invalid"""
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None

def clean_percent(s: str) -> Optional[float]:
    """'0%' → 0.0"""
    if not s:
        return None
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None

def clean_int(s: str) -> Optional[int]:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# 7. PARSERS FOR SPECIFIC SECTIONS
# --------------------------------------------------------------------------- #
def parse_summary_section(grid: List[List[str]]) -> Dict[str, Any]:
    """
    Parses the top summary block:
    Credit Score, Payments, Freeze, Alert, Deceased
    """
    data = {
        "credit_score": None,
        "monthly_payment": None,
        "credit_freeze": None,
        "fraud_alert": None,
        "deceased": None,
    }
    if not grid or len(grid) < 2:
        return data

    # First row usually has values
    row = [cell for cell in grid[0] if cell]
    if len(row) >= 5:
        data["credit_score"] = clean_int(row[0])
        data["monthly_payment"] = clean_currency(row[1].replace("/mo", ""))
        data["credit_freeze"] = row[2].lower() == "yes"
        data["fraud_alert"] = row[3].lower() == "yes"
        data["deceased"] = row[4].lower() == "yes"
    return data

def _normalize_address_line(line: str) -> str:
    # Normalize whitespace, remove stray leading numbers on their own lines,
    # and convert to Title Case for street portion while preserving city caps.
    s = " ".join([ln.strip() for ln in line.splitlines() if ln.strip() and not re.fullmatch(r"\d+", ln.strip())])
    # replace repeated whitespace and ensure commas between city/state
    s = re.sub(r"\s+", " ", s).strip()
    # if there is a pattern like 'CITY, ST ZIP' ensure it is intact; otherwise basic title-case
    parts = s.split(',')
    if len(parts) >= 2:
        street = parts[0].strip().title()
        rest = [p.strip() for p in parts[1:]]
        return street + ', ' + ', '.join(rest)
    else:
        return s.title()


def parse_personal_info(grid: List[List[str]]) -> Dict[str, Any]:
    """
    Name, Age, Addresses
    """
    data = {
        "names": [],
        "age": None,
        "addresses": [],
        # primary address convenience field
        "address": None,
    }
    name_pattern = re.compile(r"^[A-Z]+(?:\s[A-Z]+)*$")
    addr_pattern = re.compile(r"\d+.*")

    for row in grid:
        for cell in row:
            cell = cell.strip()
            if not cell:
                continue
            if name_pattern.match(cell) and len(cell.split()) >= 2:
                if cell not in data["names"]:
                    data["names"].append(cell)
            elif "Age:" in cell:
                age_match = re.search(r"Age:\s*(\d+)", cell)
                if age_match:
                    data["age"] = int(age_match.group(1))
            elif addr_pattern.match(cell):
                norm = _normalize_address_line(cell)
                if norm not in data["addresses"]:
                    data["addresses"].append(norm)

    # derive a sensible primary address: prefer one that contains a comma and a zip code
    def _looks_like_full_address(a: str) -> bool:
        return bool(re.search(r"\d+\s+\w+", a)) and (',' in a and re.search(r"\d{5}", a))

    primary = None
    for a in data["addresses"]:
        if _looks_like_full_address(a):
            primary = a
            break
    if not primary and data["addresses"]:
        primary = data["addresses"][0]

    data["address"] = primary
    return data

def parse_category_summary(grid: List[List[str]]) -> Dict[str, Any]:
    """
    Revolving Accounts (Open): 1 / $0
    """
    data: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"(.+?)\s*\(Open\)\s*(\d+)\s*/\s*\$([\d,]+)", re.I)

    for row in grid:
        for cell in row:
            cell = cell.strip()
            match = pattern.search(cell)
            if match:
                category = match.group(1).strip().lower().replace(" ", "_")
                count = int(match.group(2))
                balance = clean_currency("$" + match.group(3))
                data[category] = {"count": count, "balance": balance}
    return data

def parse_credit_alerts(grid: List[List[str]]) -> Dict[str, Any]:
    data = {
        "public_records": 0,
        "collections_open": 0,
        "collections_closed": 0,
        "inquiries_last_6_months": 0,
        "late_pays_last_2_years": 0,
        "late_pays_last_2_plus_years": 0,
    }
    for row in grid:
        text = " ".join(row).lower()
        if "public records" in text:
            data["public_records"] = _extract_number_after(text, "records")
        elif "collections" in text and "open" in text and "closed" in text:
            m = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if m:
                data["collections_open"] = int(m.group(1))
                data["collections_closed"] = int(m.group(2))
        elif "inquiries" in text and "6 months" in text:
            data["inquiries_last_6_months"] = _extract_number_after(text, "months")
        elif "late pays" in text and "2/2+ years" in text:
            m = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if m:
                data["late_pays_last_2_years"] = int(m.group(1))
                data["late_pays_last_2_plus_years"] = int(m.group(2))
    return data

def _extract_number_after(text: str, keyword: str) -> int:
    m = re.search(rf"{keyword}\s*(\d+)", text)
    return int(m.group(1)) if m else 0

def parse_credit_factors(grid: List[List[str]]) -> List[str]:
    factors = []
    for row in grid:
        for cell in row:
            cell = cell.strip()
            if cell and len(cell) > 3:
                factors.append(cell)
    return factors

def parse_account_table(grid: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Parses Open/Closed account tables.
    Handles headers and subtables (Revolving, Installment, etc.)
    """
    accounts: List[Dict[str, Any]] = []
    current_section = None
    headers = None

    for i, row in enumerate(grid):
        if not any(row):
            continue

        # Detect section header (e.g., "Revolving Accounts")
        if len(row) == 1 and row[0].strip():
            section = row[0].strip().lower()
            if any(kw in section for kw in ["revolving", "installment", "real estate", "line of credit", "miscellaneous"]):
                current_section = section
                continue

        # Detect "No ... Accounts"
        if any("no " in cell.lower() and "account" in cell.lower() for cell in row):
            continue

        # Detect column headers
        if "balance" in "".join(row).lower() and "limit" in "".join(row).lower():
            headers = [h.lower().replace(" ", "_").replace("%", "pct") for h in row if h]
            continue

        # Detect totals row
        if headers and "total" in "".join(row).lower():
            continue

        # Parse account row
        if headers and len(row) >= len(headers):
            acct = {}
            for h, val in zip(headers, row):
                val = val.strip()
                if h == "balance":
                    acct[h] = clean_currency(val)
                elif h == "limit":
                    acct[h] = clean_currency(val)
                elif h == "pct":
                    acct["utilization_pct"] = clean_percent(val)
                elif h == "payment_resp":
                    acct[h] = val
                elif h == "age":
                    acct[h] = clean_int(val) or val
                else:
                    acct[h] = val
            if acct.get("balance") is not None or acct.get("limit"):
                acct["section"] = current_section
                accounts.append(acct)

    return accounts

# --------------------------------------------------------------------------- #
# 8. MAIN PIPELINE
# --------------------------------------------------------------------------- #
def extract_credit_report(pdf_path: Path, output_json: Path) -> None:
    pdf = pdfplumber.open(pdf_path)
    full_report: Dict[str, Any] = {
        "metadata": {
            "source_file": pdf_path.name,
            "extraction_date": "2025-10-31",
            "json_version": JSON_VERSION,
        },
        "summary": {},
        "personal": {},
        "categories": {},
        "alerts": {},
        "factors": [],
        "open_accounts": [],
        "closed_accounts": [],
    }

    all_tables: List[Tuple[int, Dict]] = []  # (page_idx, table_info)

    # Step 1: Detect tables across all pages
    for page_idx in range(len(pdf.pages)):
        print(f"Processing page {page_idx + 1}/{len(pdf.pages)}...", file=sys.stderr)
        img = pdf_page_to_image(pdf_path, page_idx + 1)
        tables = detect_tables_in_page(img)
        for table in tables:
            all_tables.append((page_idx, table))

    print(f"Detected {len(all_tables)} table(s).", file=sys.stderr)

    # Step 2: Extract and classify each table
    for page_idx, table in all_tables:
        bbox = table["bbox"]
        img = pdf_page_to_image(pdf_path, page_idx + 1)
        cropped_img = crop_table_image(img, bbox)

        # Use pdfplumber on original PDF with scaled bbox
        grid = extract_table_grid(pdf, page_idx, bbox)

        if not grid:
            continue

        # Heuristic classification based on content
        text_sample = " ".join(" ".join(row) for row in grid[:3]).lower()

        if "credit score" in text_sample or "/mo" in text_sample:
            full_report["summary"].update(parse_summary_section(grid))
        elif "name:" in text_sample or "age:" in text_sample:
            full_report["personal"].update(parse_personal_info(grid))
        elif "revolving accounts (open)" in text_sample:
            full_report["categories"].update(parse_category_summary(grid))
        elif "public records" in text_sample or "collections" in text_sample:
            full_report["alerts"].update(parse_credit_alerts(grid))
        elif any(f in text_sample for f in ["no open 1k+", "too few", "avg age"]):
            full_report["factors"].extend(parse_credit_factors(grid))
        elif "open accounts" in text_sample:
            full_report["open_accounts"].extend(parse_account_table(grid))
        elif "closed accounts" in text_sample:
            full_report["closed_accounts"].extend(parse_account_table(grid))

    # Step 3: Write JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Extraction complete → {output_json}", file=sys.stderr)

# --------------------------------------------------------------------------- #
# 9. CLI ENTRYPOINT
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 3:
        print("Usage: python credit_extract.py <input.pdf> <output.json>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    if not pdf_path.is_file():
        print(f"Error: {pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        extract_credit_report(pdf_path, json_path)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()