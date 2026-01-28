
import pytest
import pandas as pd
import json
from pathlib import Path
from src.eval_algos import models, param_distributions
from src.pipeline_coordinator import MLPipelineCoordinator
from src.utils import load_column_headers

def make_df_with_columns(cols):
    # Generate uncorrelated data for each column
    import numpy as np
    rng = np.random.default_rng(42)
    # Load schema to determine which columns are categorical, and generate all use_RF columns
    with open('src/column_headers.json', 'r', encoding='utf-8') as f:
        header_data = json.load(f)
    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {'true', '1', 'yes', 'y'}
        if isinstance(val, int):
            return val == 1
        return False
    use_rf_cols = [col for col in header_data if is_true(col.get('use_RF', False))]
    col_types = {col['name']: col.get('categorical', 'False') == 'True' for col in use_rf_cols}
    all_rf_colnames = [col['name'] for col in use_rf_cols]
    print(f"[DEBUG] Generating columns: {all_rf_colnames}")
    data = {}
    for col in all_rf_colnames:
        if col_types.get(col, False):
            # categorical: use alternating integer values (0, 1)
            data[col] = [i % 2 for i in range(4)]
        else:
            # numeric: use random ints
            data[col] = rng.integers(0, 100, size=4)
    # Only return the columns requested by the test
    df = pd.DataFrame(data)
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    return df[[c for c in cols if c in df.columns]]

def test_classifier_type_column_selection():
    # Load schema columns
    with open('src/column_headers.json', 'r', encoding='utf-8') as f:
        header_data = json.load(f)
    # For each classifier type, ensure correct columns are selected
    for clf_name in models:
        def infer_classifier_type(model_name: str) -> str:
            name = model_name.lower()
            if 'xgb' in name:
                return 'XGB'
            if 'cat' in name:
                return 'Cat'
            if 'lgbm' in name or 'lightgbm' in name:
                return 'LGBM'
            if 'tree' in name:
                return 'Tree'
            if 'linear' in name or 'logistic' in name:
                return 'Linear'
            if 'nn' in name or 'mlp' in name or 'neural' in name:
                return 'NN'
            if 'nb' in name or 'naive' in name:
                return 'NB'
            if 'svm' in name:
                return 'SVM'
            if 'knn' in name:
                return 'KNN'
            return model_name
        classifier_type = infer_classifier_type(clf_name)
        use_flag = f'use_{classifier_type}'
        # Accept both string and boolean for use_* flags
        def is_true(val):
            return val is True or (isinstance(val, str) and val.lower() == 'true')
        expected_features = [col['name'] for col in header_data if col.get('X') == 'True' and is_true(col.get(use_flag, False))]
        expected_categoricals = [col['name'] for col in header_data if col.get('categorical') == 'True' and is_true(col.get(use_flag, False))]
        # If no columns are marked for this classifier_type, skip test for this classifier
        if not expected_features and not expected_categoricals:
            continue
        all_cols = [col['name'] for col in header_data]
        df = make_df_with_columns(all_cols)
        headers = load_column_headers(Path('src/column_headers.json'), df, classifier_type=classifier_type)
        assert set(headers['feature_cols']) == set(expected_features), f"Features for {clf_name} ({classifier_type}) do not match schema"
        assert set(headers['categorical_cols']) == set(expected_categoricals), f"Categoricals for {clf_name} ({classifier_type}) do not match schema"

def test_pipeline_coordinator_search_models_classifier_type():
    # Integration: search_models should not error and should use correct columns
    import numpy as np
    from itertools import islice
    with open('src/column_headers.json', 'r', encoding='utf-8') as f:
        header_data = json.load(f)
    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {'true', '1', 'yes', 'y'}
        if isinstance(val, int):
            return val == 1
        return False
    # Collect all columns required for use_RF: true (features, categoricals, and targets)
    rf_features = [col['name'] for col in header_data if col.get('X') == 'True' and is_true(col.get('use_RF', False))]
    rf_categoricals = [col['name'] for col in header_data if col.get('categorical') == 'True' and is_true(col.get('use_RF', False))]
    rf_targets = [col['name'] for col in header_data if col.get('Y') == 'True' and is_true(col.get('use_RF', False))]
    used_cols = list(set(rf_features) | set(rf_categoricals) | set(rf_targets))
    print(f"[DEBUG] use_RF features: {rf_features}")
    print(f"[DEBUG] use_RF categoricals: {rf_categoricals}")
    print(f"[DEBUG] use_RF targets: {rf_targets}")
    print(f"[DEBUG] used_cols: {used_cols}")
    if not used_cols:
        pytest.skip("No columns with use_RF: true in schema; skipping integration test.")
    subset = dict(islice(models.items(), 2))
    param_dist = {k: param_distributions[k] for k in subset}
    df = make_df_with_columns(used_cols)
    print(f"[DEBUG] df columns: {list(df.columns)}")
    # Compute expected features/categoricals for the first classifier in subset
    def infer_classifier_type(model_name: str) -> str:
        name = model_name.lower()
        if 'xgb' in name:
            return 'XGB'
        if 'cat' in name:
            return 'Cat'
        if 'lgbm' in name or 'lightgbm' in name:
            return 'LGBM'
        if 'tree' in name:
            return 'Tree'
        if 'linear' in name or 'logistic' in name:
            return 'Linear'
        if 'nn' in name or 'mlp' in name or 'neural' in name:
            return 'NN'
        if 'nb' in name or 'naive' in name:
            return 'NB'
        if 'svm' in name:
            return 'SVM'
        if 'knn' in name:
            return 'KNN'
        if 'rf' in name or 'randomforest' in name:
            return 'RF'
        return model_name
    first_clf = next(iter(subset))
    classifier_type = infer_classifier_type(first_clf)
    use_flag = f'use_{classifier_type}'
    print(f"[DEBUG] classifier_type: {classifier_type}, use_flag: {use_flag}")
    for col in header_data:
        print(f"[DEBUG] {col['name']} | {use_flag} = {col.get(use_flag, None)} | X = {col.get('X', None)} | categorical = {col.get('categorical', None)}")
    # Use the robust is_true already defined above
    print(f"[DEBUG] classifier_type: {classifier_type}, use_flag: {use_flag}")
    for col in header_data:
        print(f"[DEBUG] {col['name']} | {use_flag} = {col.get(use_flag, None)} | X = {col.get('X', None)} | categorical = {col.get('categorical', None)}")
    expected_features = [col['name'] for col in header_data if col.get('X') == 'True' and is_true(col.get(use_flag, False))]
    expected_categoricals = [col['name'] for col in header_data if col.get('categorical') == 'True' and is_true(col.get(use_flag, False))]
    relevant_cols = list(set(expected_features) | set(expected_categoricals))
    print(f"[DEBUG] relevant_cols: {relevant_cols}")
    df_relevant = df[relevant_cols]
    print(f"[DEBUG] df_relevant columns: {list(df_relevant.columns)}")
    y = np.array([0, 1, 0, 1])
    coordinator = MLPipelineCoordinator(enable_debugging=False, export_debug_info=False)
    # Set cv=2 to avoid n_splits > n_samples error
    top, best = coordinator.search_models(subset, param_dist, df_relevant, y, n_top=1, smoke=True, column_headers_json=Path('src/column_headers.json'), cv=2)
    assert isinstance(top, list)
    assert isinstance(best, dict)
