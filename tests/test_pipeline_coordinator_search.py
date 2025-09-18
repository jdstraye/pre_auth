import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure `src/` is on sys.path so imports like `pipeline_coordinator` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_coordinator import MLPipelineCoordinator
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from utils import gv


def make_synthetic_df(n_samples=80, n_features=5, random_state=gv.RANDOM_STATE):
    Xnp, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=3, n_redundant=0,
                                 n_classes=2, random_state=random_state)
    X = pd.DataFrame(Xnp, columns=[f"f{i}" for i in range(n_features)])
    return X, y


def test_search_models_smoke_runs_and_returns_candidates():
    coord = MLPipelineCoordinator(enable_debugging=False, export_debug_info=False)
    X, y = make_synthetic_df()

    models = {
        'RandomForest': RandomForestClassifier(random_state=gv.RANDOM_STATE, n_estimators=5)
    }

    param_distributions = {
        'RandomForest': {
            'feature_selecting_classifier__max_features': [2, 3],
            'smote__enabled': [False]
        }
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=gv.RANDOM_STATE)

    top_candidates, best_summary = coord.search_models(models, param_distributions, X, y, n_top=2, random_search_mult=0.05, smoke=True, cv=cv, n_jobs=1)

    assert isinstance(top_candidates, list)
    assert len(top_candidates) > 0
    assert isinstance(best_summary, dict)
    assert 'model' in best_summary and 'score' in best_summary


def test_search_models_target_f1_reached():
    coord = MLPipelineCoordinator(enable_debugging=False, export_debug_info=False)
    # create a larger, easier classification problem to reach high f1
    X, y = make_synthetic_df(n_samples=400, n_features=6)

    models = {
        'RandomForest': RandomForestClassifier(random_state=gv.RANDOM_STATE, n_estimators=100)
    }

    param_distributions = {
        'RandomForest': {
            'feature_selecting_classifier__max_features': [6],
            'smote__enabled': [False]
        }
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=gv.RANDOM_STATE)

    # Use target_f1 that a well-fit RF should attain on this synthetic dataset
    target_f1 = 0.7
    top_candidates, best_summary = coord.search_models(models, param_distributions, X, y, n_top=1, random_search_mult=1.0, smoke=False, cv=cv, n_jobs=1, target_f1=target_f1)

    assert isinstance(top_candidates, list)
    assert len(top_candidates) >= 1
    assert isinstance(best_summary, dict)
    assert best_summary.get('score', 0.0) >= target_f1
