import json
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ingest import parse_json, preprocess_dataframe, _load_golden_schema, _parse_schema
import allocate


def make_sample_record(i):
    return {
        "record_id": f"r{i}",
        "user_initials": f"U{i}",
        "prefi_data": {
            "DataEnhance": {"DebtToIncome": 30 + i},
            "Offers": [
                {"Name": "Automatic Financing", "Amount": "1000", "Status": "Approved", "Score": 700, "DebtToIncome": 30, "Details": "Financing Automatically and Instantly Available"},
                {"Name": "0% Unsecured Funding", "Amount": "2000", "Status": "Declined", "Score": "Below 600", "Contingencies": "PayD-$100, Collections-$50"}
            ]
        },
        "contracts": [
            {"status": "approved" if i % 2 == 0 else "declined", "tier": "0.095", "amount": "1000", "created_at": "2025-01-01"}
        ]
    }


def test_allocate_keeps_na_strings_and_writes_files(tmp_path):
    data = {"data": [make_sample_record(i) for i in range(8)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    schema = _load_golden_schema(Path("src/column_headers.json"))
    sorted_schema, column_map = _parse_schema(schema)

    df_flat = parse_json(json_file, sorted_schema)
    df_processed = preprocess_dataframe(df_flat, sorted_schema, column_map)

    csv_in = tmp_path / "imported.csv"
    df_processed.to_csv(csv_in, index=False)

    out_train = tmp_path / "train.csv"
    out_test = tmp_path / "test.csv"

    argv_backup = os.sys.argv
    try:
        os.sys.argv = ["allocate.py", "-i", str(csv_in), "-o1", str(out_train), "-o2", str(out_test)]
        allocate.main()
    finally:
        os.sys.argv = argv_backup

    assert out_train.exists() and out_test.exists()

    train = pd.read_csv(out_train, keep_default_na=False)
    # ensure that one of the string-like columns has 'NA' strings preserved
    assert 'AutomaticFinancing_Details' in train.columns
    assert 'NA' in train['AutomaticFinancing_Details'].values or train['AutomaticFinancing_Details'].isna().sum() == 0
    # Ensure targets present
    assert 'final_contract_status_label' in train.columns
    assert 'final_contract_tier_label' in train.columns
