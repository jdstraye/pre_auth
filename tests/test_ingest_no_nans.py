import pandas as pd
import numpy as np
from src.ingest import ensure_no_nans
from src.ingest import preprocess_dataframe


def test_ensure_no_nans_replaces_values():
    # Minimal schema: one numeric X, one OHE, one string-like
    schema = [
        {"name": "num_feature", "X": "True", "categorical": "False"},
        {"name": "cat_flag", "ohe": {"Yes": "cat_flag"}},
        {"name": "desc_Status"}
    ]
    df = pd.DataFrame({
        "num_feature": [1.0, np.nan, np.inf],
        "cat_flag": [1, None, 0],
        "desc_Status": ["ok", None, "ok"]
    })

    cleaned = ensure_no_nans(df, schema)

    # Numeric NaN/inf replaced with -1
    assert cleaned.loc[1, "num_feature"] == -1
    assert cleaned.loc[2, "num_feature"] == -1

    # OHE NaN replaced with 0 and ints
    assert cleaned["cat_flag"].dtype.kind in ("i", "u")
    assert cleaned.loc[1, "cat_flag"] == 0

    # Object NaN replaced with 'NA'
    assert cleaned.loc[1, "desc_Status"] == 'NA'


def test_preprocess_dataframe_cleans_nans():
    # Build a minimal schema and DataFrame that would normally be processed
    schema = [
        {"name": "A", "X": "True", "categorical": "False"},
        {"name": "B", "ohe": {"X": "B_X"}},
        {"name": "B_X", "ohe_from": "B", "ohe_key": "X"}
    ]
    df = pd.DataFrame({"A": [1, None, float('inf')], "B": ['X', None, 'X']})
    sorted_schema, column_map = (schema, {c['name']: c for c in schema})
    processed = preprocess_dataframe(df, sorted_schema, column_map)

    # After preprocessing, numeric inf/NaN should be replaced
    assert processed['A'].isnull().sum() == 0
    # OHE column should exist and be integer 0/1 and have no NaNs
    assert 'B_X' in processed.columns
    assert processed['B_X'].isnull().sum() == 0
