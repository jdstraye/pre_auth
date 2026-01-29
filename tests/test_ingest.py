import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ingest import parse_json, preprocess_dataframe, _load_golden_schema, _parse_schema
from src.utils import load_column_headers


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


def test_preprocess_has_no_nan_and_ohe_int(tmp_path):
    data = {"data": [make_sample_record(i) for i in range(6)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    schema = _load_golden_schema(Path("src/column_headers.json"))
    sorted_schema, column_map = _parse_schema(schema)

    df_flat = parse_json(json_file, sorted_schema)
    df_processed = preprocess_dataframe(df_flat, sorted_schema, column_map)

    headers = load_column_headers(Path("src/column_headers.json"), df_processed)
    feature_cols = headers['feature_cols']
    ohe_cols = headers['ohe_cols']

    # No NaNs in feature columns
    nan_sum = df_processed[feature_cols].isna().sum().sum()
    assert nan_sum == 0, f"Found NaNs in features: {nan_sum}"

    # OHE columns should be ints and only 0/1 values
    for c in ohe_cols:
        assert c in df_processed.columns, f"OHE col missing: {c}"
        assert pd.api.types.is_integer_dtype(df_processed[c].dtype), f"OHE col not int: {c} ({df_processed[c].dtype})"
        vals = set(df_processed[c].unique().tolist())
        assert vals.issubset({0, 1}), f"OHE col has non-binary values: {c} -> {vals}"
