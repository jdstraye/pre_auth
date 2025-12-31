import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_coordinator import MLPipelineCoordinator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import gv


def make_large_classification(n_samples=1000, n_features=20):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=10,
                               n_redundant=2, n_classes=2, weights=[0.8, 0.2], random_state=gv.RANDOM_STATE)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return X_df, y


def test_search_large_param_space_and_evaluate_metrics():
    """Run a reasonably large hyperparameter search and evaluate performance metrics for top candidates."""
    # Build dataset and split into train/test (80/20)
    X, y = make_large_classification(n_samples=800, n_features=16)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=gv.RANDOM_STATE)

    # Coordinator with debugging off
    coord = MLPipelineCoordinator(enable_debugging=False, export_debug_info=False)

    # Models and param distributions (moderately large grid)
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=gv.RANDOM_STATE, n_estimators=50),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=gv.RANDOM_STATE, n_estimators=50)
    }

    param_distributions = {
        'RandomForestClassifier': {
            'smote__enabled': [False, True],
            'smote__k_neighbors': [3, 5],
            'feature_selecting_classifier__max_features': [5, 8, None],
            'feature_selecting_classifier__estimator__n_estimators': [50, 100],
            'feature_selecting_classifier__estimator__max_depth': [None, 10]
        },
        'ExtraTreesClassifier': {
            'smote__enabled': [False, True],
            'smote__k_neighbors': [3, 5],
            'feature_selecting_classifier__max_features': [5, 8, None],
            'feature_selecting_classifier__estimator__n_estimators': [50, 100],
            'feature_selecting_classifier__estimator__max_depth': [None, 10]
        }
    }

    # Create a 3-fold CV for thoroughness
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=gv.RANDOM_STATE)

    # Run search with a multiplier to bring more samples, but not too many.
    top_candidates, best_summary = coord.search_models(models=models,
                                                      param_distributions=param_distributions,
                                                      X=X_train,
                                                      y=y_train,
                                                      n_top=5,
                                                      random_search_mult=1.0,
                                                      smoke=False,
                                                      cv=cv,
                                                      n_jobs=1)

    assert isinstance(top_candidates, list) and len(top_candidates) > 0
    assert isinstance(best_summary, dict)

    # Evaluate metrics for top candidates on the held-out test set
    metrics_rows = []
    for model_name, params, score in top_candidates:
        base_estimator = models[model_name]
        # Create pipeline with given params
        smote_enabled = bool(params.get('smote__enabled', True))
        smote_k = int(params.get('smote__k_neighbors', 3))
        smote_cats = []
        from components.smote_sampler import MaybeSMOTESampler
        from components.feature_selector import FeatureSelectingClassifier
        smote_cfg = {"enabled": smote_enabled, "categorical_feature_names": smote_cats, "k_neighbors": smote_k}
        pipeline = coord.create_pipeline(base_estimator=base_estimator, smote_config=smote_cfg,
                                         feature_selection_config={"max_features": params.get('feature_selecting_classifier__max_features', None),
                                                                   "threshold": params.get('feature_selecting_classifier__threshold', None)})
        # set nested params onto pipeline if present
        try:
            pipeline.set_params(**params)
        except Exception:
            pass
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        # compute metrics
        metrics = {
            'model': model_name,
            'params': params,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred))
        }
        metrics_rows.append(metrics)

    # Verify metrics are sensible
    assert len(metrics_rows) == len(top_candidates)
    for r in metrics_rows:
        assert 0.0 <= r['accuracy'] <= 1.0
        assert 0.0 <= r['precision'] <= 1.0
        assert 0.0 <= r['recall'] <= 1.0
        assert 0.0 <= r['f1'] <= 1.0

    # Optionally print results for manual inspection
    df_metrics = pd.DataFrame(metrics_rows)
    print(df_metrics.to_string())
