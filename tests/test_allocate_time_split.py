import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_time_based_split(tmp_path, monkeypatch):
    # create sample dataframe
    n = 300
    base = datetime(2024,1,1)
    rows = []
    for i in range(n):
        rows.append({
            'record_id': f'r{i}',
            'user_initials': 'AA',
            'final_contract_status_label': 1 if i % 3 == 0 else 0,
            'final_contract_tier_label': 0 if i % 2 == 0 else 1,
            'final_contract_created_at': (base + timedelta(days=i)).isoformat()
        })
    df = pd.DataFrame(rows)
    in_path = tmp_path / 'input.csv'
    out_train = tmp_path / 'train.csv'
    out_test = tmp_path / 'test.csv'
    df.to_csv(in_path, index=False)

    # Run allocator with time strategy
    monkeypatch.setattr(sys, 'argv', ['allocate.py', '-i', str(in_path), '-o1', str(out_train), '-o2', str(out_test), '--strategy', 'time', '--time_col', 'final_contract_created_at'])
    import src.allocate as allocate
    allocate.main()

    train = pd.read_csv(out_train)
    test = pd.read_csv(out_test)
    train['final_contract_created_at'] = pd.to_datetime(train['final_contract_created_at'], errors='coerce')
    test['final_contract_created_at'] = pd.to_datetime(test['final_contract_created_at'], errors='coerce')

    assert train['final_contract_created_at'].max() <= test['final_contract_created_at'].min()
