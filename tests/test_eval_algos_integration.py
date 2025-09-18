import subprocess
from pathlib import Path
import time
import json
import pandas as pd


def test_eval_algos_smoke_run(tmp_path):
    # Create minimal synthetic CSV with all features & targets from schema and run eval_algos
    schema = Path("src/column_headers.json")
    with open(schema, 'r', encoding='utf8') as f:
        header_data = json.load(f)
    feature_cols = [h['name'] for h in header_data if h.get('X') == 'True']
    # ensure targets included
    target_cols = [h['name'] for h in header_data if h.get('Y') == 'True']
    cols = feature_cols + target_cols
    n = 120
    df = pd.DataFrame(index=range(n), columns=cols)
    import numpy as np
    for c in feature_cols:
        if c.endswith('_Approved') or c.endswith('_Declined') or c.endswith('_NA_'):
            df[c] = np.random.randint(0, 2, size=n)
        else:
            df[c] = np.random.normal(size=n)
    # Construct targets: alternating statuses/tiers
    df[target_cols[0]] = np.tile([0, 1], int(np.ceil(n/2)))[:n]
    df[target_cols[1]] = np.tile([0, 1, 2], int(np.ceil(n/3)))[:n]

    train_csv = tmp_path / 'train.csv'
    test_csv = tmp_path / 'test.csv'
    df.iloc[:80].to_csv(train_csv, index=False)
    df.iloc[80:].to_csv(test_csv, index=False)
    cmd = ["python", "-m", "src.eval_algos", "--train_csv", str(train_csv), "--test_csv", str(test_csv), "--column_headers_json", str(schema), "--random_search_mult", "0.02", "--use_coordinator"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr


def test_eval_algos_smoke_run_with_target_f1(tmp_path):
    schema = Path("src/column_headers.json")
    with open(schema, 'r', encoding='utf8') as f:
        header_data = json.load(f)
    feature_cols = [h['name'] for h in header_data if h.get('X') == 'True']
    target_cols = [h['name'] for h in header_data if h.get('Y') == 'True']
    cols = feature_cols + target_cols
    n = 120
    df = pd.DataFrame(index=range(n), columns=cols)
    import numpy as np
    for c in feature_cols:
        if c.endswith('_Approved') or c.endswith('_Declined') or c.endswith('_NA_'):
            df[c] = np.random.randint(0, 2, size=n)
        else:
            df[c] = np.random.normal(size=n)
    df[target_cols[0]] = np.tile([0, 1], int(np.ceil(n/2)))[:n]
    df[target_cols[1]] = np.tile([0, 1, 2], int(np.ceil(n/3)))[:n]

    train_csv = tmp_path / 'train.csv'
    test_csv = tmp_path / 'test.csv'
    df.iloc[:80].to_csv(train_csv, index=False)
    df.iloc[80:].to_csv(test_csv, index=False)
    cmd = ["python", "-m", "src.eval_algos", "--train_csv", str(train_csv), "--test_csv", str(test_csv), "--column_headers_json", str(schema), "--random_search_mult", "0.02", "--use_coordinator", "--target_f1", "0.55"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
