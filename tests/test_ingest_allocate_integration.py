import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure src is importable and components package resolved
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ingest import flatten_weaviate_data, preprocess_dataframe, _load_golden_schema, _parse_schema
import allocate
from pipeline_coordinator import MLPipelineCoordinator
from sklearn.ensemble import RandomForestClassifier


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


def test_ingest_allocate_pipeline_smoke(tmp_path):
    # Create minimal JSON file
    data = {"data": [make_sample_record(i) for i in range(12)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    # Load schema and parse
    schema = _load_golden_schema(Path("src/column_headers.json"))
    sorted_schema, column_map = _parse_schema(schema)

    # Run ingest functions
    df_flat = flatten_weaviate_data(json_file, sorted_schema)
    df_processed = preprocess_dataframe(df_flat, sorted_schema, column_map)

    # Save to CSV for allocate
    csv_in = tmp_path / "imported.csv"
    df_processed.to_csv(csv_in, index=False)

    # Run allocate.main by constructing args
    out_train = tmp_path / "train.csv"
    out_test = tmp_path / "test.csv"
    # call allocate via its main using environment args parsing
    argv_backup = os.sys.argv
    try:
        os.sys.argv = ["allocate.py", "-i", str(csv_in), "-o1", str(out_train), "-o2", str(out_test)]
        allocate.main()
    finally:
        os.sys.argv = argv_backup

    assert out_train.exists() and out_test.exists()

    # Read train and run pipeline coordinator on it
    train = pd.read_csv(out_train, keep_default_na=False)
    
    # Use schema to select feature columns and target columns to avoid accidentally including raw string columns
    from src.utils import load_column_headers
    headers = load_column_headers(Path("src/column_headers.json"), train)
    feature_cols = headers['feature_cols']
    target_cols = headers['target_cols']
    y = train['final_contract_status_label'].to_numpy()
    X = train[feature_cols]

    coord = MLPipelineCoordinator()
    report = coord.validate_pipeline_input(X, y)
    assert isinstance(report, dict)
    # Expect validation to pass since preprocess populates sentinel strings and allocate preserves them
    assert report.get("passed") is True

    # Create pipeline with SMOTE disabled for speed and fit
    pipeline = coord.create_pipeline(RandomForestClassifier(random_state=42, n_estimators=5), smote_config={"enabled": False}, feature_selection_config={"max_features": 5, "threshold": None})
    # Fit pipeline while skipping input validation (we're testing end-to-end fit behavior)
    fitted = coord.fit_pipeline(pipeline, X, y, validate_input=False)
    assert fitted is not None

    # Now call eval_algos with a minimal hyperparameter grid to ensure it runs end-to-end
    from src import eval_algos
    orig_params = eval_algos.param_distributions
    try:
        eval_algos.param_distributions = {
            'RandomForestClassifier': {
                'smote__enabled': [False],
                'smote__k_neighbors': [1],
                'feature_selecting_classifier__max_features': [5],
                'feature_selecting_classifier__threshold': [None],
                'feature_selecting_classifier__estimator__n_estimators': [10],
                'feature_selecting_classifier__estimator__max_depth': [3]
            }
        }
        import argparse
        args = argparse.Namespace(train_csv=str(out_train), test_csv=str(out_test), column_headers_json=Path('src/column_headers.json'), random_search_mult=0.02, use_coordinator=True)
        eval_algos.main(args)
    finally:
        eval_algos.param_distributions = orig_params
