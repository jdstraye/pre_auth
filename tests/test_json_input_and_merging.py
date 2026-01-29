import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.ingest import parse_json, preprocess_dataframe, _load_golden_schema, _parse_schema
import src.run_exhaustive_search as run_exhaustive_search
import scripts.run_intensive_search as run_intensive_search_script
import src.eval_algos as eval_algos


def make_sample_record(i):
    return {
        "record_id": f"r{i}",
        "user_initials": f"U{i}",
        "prefi_data": {
            "DataEnhance": {"DebtToIncome": 30 + i},
            "Offers": [
                {"Name": "Automatic Financing", "Amount": "1000", "Status": "Approved", "Score": 700, "DebtToIncome": 30, "Details": "Financing"},
                {"Name": "0% Unsecured Funding", "Amount": "2000", "Status": "Declined", "Score": "Below 600", "Contingencies": "PayD-$100, Collections-$50"}
            ]
        },
        "contracts": [
            {"status": "approved" if i % 2 == 0 else "declined", "tier": "0.095", "amount": "1000", "created_at": "2025-01-01"}
        ]
    }


def test_run_exhaustive_accepts_input_json(tmp_path, monkeypatch):
    # Small JSON
    data = {"data": [make_sample_record(i) for i in range(8)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    # Small param grid and single dummy model to keep runtime tiny
    monkeypatch.setattr(run_exhaustive_search, "eval_param_distributions", {"Dummy": {"smote__enabled": [False]}})
    monkeypatch.setattr(run_exhaustive_search, "eval_models", {"Dummy": DummyClassifier(strategy='most_frequent')})

    outdir = tmp_path / "out"
    args = SimpleNamespace(
        input_json=json_file,
        data_csv=None,
        column_headers=Path("src/column_headers.json"),
        output_dir=outdir,
        n_samples_per_model=1,
        flush_every=1,
        cv=2,
        limit=1,
        exhaustive=False,
        no_progress=True,
        debug=False
    )

    run_exhaustive_search.run_search(args)
    out_csv = outdir / 'search_results.csv'
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert 'mean_f1' in df.columns


def test_run_intensive_accepts_input_json(tmp_path):
    data = {"data": [make_sample_record(i) for i in range(8)]}
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps(data))

    # Temporarily shrink eval_algos grid to speed up intensive run
    orig = eval_algos.param_distributions
    try:
        eval_algos.param_distributions = {
            'Dummy': {
                'smote__enabled': [False]
            }
        }
        # Prepare argv for run_intensive_search
        argv_backup = sys.argv
        try:
            sys.argv = ["run_intensive_search.py", "--input-json", str(json_file), "--limit", "1", "--n-samples-per-model", "1", "--debug"]
            run_intensive_search_script.main()
        finally:
            sys.argv = argv_backup
        # Check output dir
        outdir = Path('models') / 'intensive_search'
        out_csv = outdir / 'search_results.csv'
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert 'mean_f1' in df.columns
    finally:
        eval_algos.param_distributions = orig


def test_score_merging_use_median(tmp_path):
    # Create a small dataframe with AF and OUF scores and edge values
    rows = [
        {"record_id": "r0", "user_initials": "U0", "final_contract_status": "approved", "final_contract_tier": "0.095", "final_contract_amount": 1000.0, "DebtToIncome": 30, "AutomaticFinancing_Score": 700, "0UnsecuredFunding_Score": 650},
        {"record_id": "r1", "user_initials": "U1", "final_contract_status": "approved", "final_contract_tier": "0.095", "final_contract_amount": 1000.0, "DebtToIncome": 31, "AutomaticFinancing_Score": None, "0UnsecuredFunding_Score": 600},
        {"record_id": "r2", "user_initials": "U2", "final_contract_status": "declined", "final_contract_tier": "0.095", "final_contract_amount": 1000.0, "DebtToIncome": 32, "AutomaticFinancing_Score": -1, "0UnsecuredFunding_Score": -1},
    ]
    df = pd.DataFrame(rows)

    schema = _load_golden_schema(Path("src/column_headers.json"))
    sorted_schema, column_map = _parse_schema(schema)

    processed = preprocess_dataframe(df, sorted_schema, column_map)

    # After merging, AutomaticFinancing_Score should be median of AF and OUF (ignoring -1 sentinels)
    af_vals = processed['AutomaticFinancing_Score'].tolist()
    # medians: (700,650)=675, (nan,600)=600, (-1,-1) -> -1
    assert af_vals[0] == 675.0
    assert af_vals[1] == 600.0
    assert af_vals[2] == -1.0
