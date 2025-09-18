import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ingest import flatten_weaviate_data, preprocess_dataframe, _load_golden_schema, _parse_schema
from src.utils import load_column_headers


def make_minimal_record(i):
    return {
        "record_id": f"r{i}",
        "user_initials": f"U{i}",
        "prefi_data": {
            "DataEnhance": {"DebtToIncome": 30 + i},
            "Offers": [
                {"Name": "Automatic Financing", "Amount": "1000", "Status": "Approved", "Score": 700, "DebtToIncome": 30, "Details": "Financing Automatically and Instantly Available"}
            ]
        },
        "contracts": [
            {"status": "approved", "tier": "0.095", "amount": "1000", "created_at": "2025-01-01"}
        ]
    }


def test_schema_mismatch_detected(tmp_path):
    # Create sample processed dataframe
    data = {"data": [make_minimal_record(i) for i in range(2)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    schema = _load_golden_schema(Path("src/column_headers.json"))
    sorted_schema, column_map = _parse_schema(schema)

    # Instead of running the whole flatten process (which validates columns),
    # create a minimal DataFrame that intentionally lacks some schema columns
    df_processed = pd.DataFrame({
        'record_id': ['r0', 'r1'],
        'user_initials': ['U0', 'U1'],
        'DebtToIncome': [30, 31],
        'final_contract_amount': [1000.0, 1500.0],
        'AutomaticFinancing_Score': [700.0, 650.0],
        'final_contract_status': ['approved', 'declined']
    })

    # Load original schema JSON and intentionally change a feature in it so it doesn't exist in processed DataFrame
    schema_modified = json.loads(Path("src/column_headers.json").read_text())

    # Find the first feature with X==True and rename it to something bogus
    for item in schema_modified:
        if item.get('X') == 'True' or item.get('X') == True:
            bogus_name = 'this_column_does_not_exist'
            item['name'] = bogus_name
            # Leave other parts of schema intact
            break

    # Write the modified schema to a temp file
    schema_path = tmp_path / "column_headers_modified.json"
    schema_path.write_text(json.dumps(schema_modified))

    # Expect load_column_headers to raise ValueError due to missing feature column
    with pytest.raises(ValueError):
        load_column_headers(schema_path, df_processed)
