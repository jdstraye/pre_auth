import pytest
import pandas as pd
from pathlib import Path
from src.utils import load_column_headers
from src.eval_algos import param_distributions
from src.pipeline_coordinator import MLPipelineCoordinator

def make_df_with_columns(cols):
    # Create a DataFrame with all columns present, filled with dummy data
    data = {c: [0, 1] for c in cols}
    return pd.DataFrame(data)

# Helper to load all feature/categorical/target columns from schema
import json
def get_schema_columns(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        header_data = json.load(f)
    feature_cols = [col['name'] for col in header_data if col.get('X') == 'True']
    categorical_cols = [col['name'] for col in header_data if col.get('categorical') == 'True']
    target_cols = [col['name'] for col in header_data if col.get('Y') == 'True']
    return feature_cols, categorical_cols, target_cols

def test_load_column_headers_returns_expected_keys():
    feature_cols, categorical_cols, target_cols = get_schema_columns('src/column_headers.json')
    # Use all required columns for a valid DataFrame
    all_cols = set(feature_cols) | set(categorical_cols) | set(target_cols)
    df = make_df_with_columns(all_cols)
    headers = load_column_headers(Path('src/column_headers.json'), df)
    assert 'feature_cols' in headers
    assert 'categorical_cols' in headers
    assert 'target_cols' in headers

def test_feature_cols_match_schema():
    feature_cols, _, _ = get_schema_columns('src/column_headers.json')
    df = make_df_with_columns(feature_cols)
    headers = load_column_headers(Path('src/column_headers.json'), df)
    feature_cols_out = headers['feature_cols']
    assert isinstance(feature_cols_out, list)
    assert all(isinstance(c, str) for c in feature_cols_out)
    # Should match schema
    assert set(feature_cols_out) == set(feature_cols)

def test_categorical_cols_match_schema():
    feature_cols, categorical_cols, _ = get_schema_columns('src/column_headers.json')
    # DataFrame must include all feature columns for schema validation
    all_needed = set(feature_cols) | set(categorical_cols)
    df = make_df_with_columns(all_needed)
    headers = load_column_headers(Path('src/column_headers.json'), df)
    categorical_cols_out = headers['categorical_cols']
    assert isinstance(categorical_cols_out, list)
    assert all(isinstance(c, str) for c in categorical_cols_out)
    assert set(categorical_cols_out) == set(categorical_cols)

def test_target_cols_match_schema():
    feature_cols, _, target_cols = get_schema_columns('src/column_headers.json')
    # DataFrame must include all feature columns for schema validation
    all_needed = set(feature_cols) | set(target_cols)
    df = make_df_with_columns(all_needed)
    headers = load_column_headers(Path('src/column_headers.json'), df)
    target_cols_out = headers['target_cols']
    assert isinstance(target_cols_out, list)
    assert all(isinstance(c, str) for c in target_cols_out)
    assert set(target_cols_out) == set(target_cols)

def test_pipeline_coordinator_selects_columns_per_classifier():
    # For each classifier, ensure pipeline can be created with correct columns
    from src.eval_algos import models
    feature_cols, categorical_cols, target_cols = get_schema_columns('src/column_headers.json')
    all_cols = set(feature_cols) | set(categorical_cols) | set(target_cols)
    df = make_df_with_columns(all_cols)
    headers = load_column_headers(Path('src/column_headers.json'), df)
    for clf_name, clf in models.items():
        coordinator = MLPipelineCoordinator()
        # This should not raise
        pipeline = coordinator.create_pipeline(base_estimator=clf, smote_config={}, feature_selection_config={})
        assert pipeline is not None
