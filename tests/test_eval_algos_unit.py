import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from src.eval_algos import get_top_models, param_distributions as orig_param_distributions
from src.utils import load_column_headers
import json


def test_get_top_models_minimal_search(tmp_path):
    # Create minimal synthetic dataset matching column headers
    schema_path = Path("src/column_headers.json")
    # create Dummy DF with required feature cols - parse JSON to avoid validation call
    with open(schema_path, 'r', encoding='utf8') as fh:
        header_data = json.load(fh)
    feature_cols = [h['name'] for h in header_data if h.get('X') == 'True']
    df = pd.DataFrame(columns=feature_cols)
    # synthesize features
    n = 60
    for i, col in enumerate(feature_cols):
        # keep OHE-like columns binary
        if col.endswith("_Approved") or col.endswith("_NA_") or col.endswith("_Declined") or col.endswith("_If_Fixed"):
            df[col] = np.random.randint(0, 2, size=n)
        else:
            df[col] = np.random.normal(loc=0.0, scale=1.0, size=n)

    # Target balancing binary
    y = np.tile([0, 1], int(np.ceil(n / 2)))[:n]

    # Monkeypatch param_distributions to small grid
    minimal = {
        'RandomForestClassifier': {
            'smote__enabled': [False],
            'smote__k_neighbors': [1],
            'feature_selecting_classifier__max_features': [5],
            'feature_selecting_classifier__threshold': [None],
            'feature_selecting_classifier__estimator__n_estimators': [10],
            'feature_selecting_classifier__estimator__max_depth': [3]
        }
    }

    try:
        from src import eval_algos
        eval_algos.param_distributions = minimal
        top, best, worst = get_top_models(df[feature_cols], y, n_top=1, phase="status", column_headers_json=schema_path, random_search_mult=0.05, n_jobs_cv=1)
        assert isinstance(top, list)
        assert best is not None
    finally:
        from src import eval_algos
        eval_algos.param_distributions = orig_param_distributions
