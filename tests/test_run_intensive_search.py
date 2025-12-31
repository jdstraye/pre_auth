import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def test_run_intensive_search_smoke(tmp_path, monkeypatch):
    # Build a tiny CSV dataset with a binary status label
    n = 20
    df = pd.DataFrame({
        'DebtToIncome': np.random.randint(10, 50, size=n),
        'AutomaticFinancing_Score': np.random.randint(500, 700, size=n),
        'AutomaticFinancing_Status_Approved': [1 if i % 2 == 0 else 0 for i in range(n)],
        'final_contract_status_label': [0 if i < n//2 else 1 for i in range(n)]
    })

    csv_path = tmp_path / 'small.csv'
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / 'out'
    args = [
        'run_intensive_search.py',
        '--data-csv', str(csv_path),
        '--output-dir', str(outdir),
        '--n-samples-per-model', '1',
        '--random-search-mult', '0.01',
        '--n-top', '1',
        '--flush-every', '1',
        '--save-format', 'csv',
        '--cv', '2',
        '--dry-run'
    ]
    monkeypatch.setattr(sys, 'argv', args)

    # Ensure joblib is available for import-time in the exhaustive script
    import types
    import sys as _sys
    _sys.modules.setdefault('joblib', types.SimpleNamespace(dump=lambda *a, **k: None, load=lambda *a, **k: None))
    # Run the script; it should complete without exception and write output files
    import scripts.run_intensive_search as ris
    ris.main()

    # Assert outputs exist
    assert outdir.exists()
    # search_results.csv should be present when flush happened
    assert (outdir / 'search_results.csv').exists()
