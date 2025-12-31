import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_coordinator import MLPipelineCoordinator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from utils import gv


def make_synthetic_df(n_samples=100, n_features=10):
    Xnp, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=6, n_redundant=0, n_classes=2, weights=[0.8, 0.2], random_state=gv.RANDOM_STATE)
    X = pd.DataFrame(Xnp, columns=[f"f{i}" for i in range(n_features)])
    return X, y


def test_find_best_classifier_minimal_samples():
    """Integration-style test: find best classifier with a minimal hyperparameter grid (1-2 samples).

    This uses the coordinator directly to avoid CLI subprocess overhead but tests end-to-end
    model search using a small parameter grid and a small CV fold count (2).
    """
    coord = MLPipelineCoordinator(enable_debugging=False, export_debug_info=False)
    X, y = make_synthetic_df(n_samples=120, n_features=8)

    # Minimal models & small param grid (only 2 param choices)
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=gv.RANDOM_STATE, n_estimators=10)
    }

    param_distributions = {
        'RandomForestClassifier': {
            'smote__enabled': [False],
            'feature_selecting_classifier__max_features': [3, None]
        }
    }

    # 2-fold CV for speed in smoke integration
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=gv.RANDOM_STATE)

    top_candidates, best_summary = coord.search_models(models=models,
                                                      param_distributions=param_distributions,
                                                      X=X,
                                                      y=y,
                                                      n_top=2,
                                                      random_search_mult=0.02,
                                                      smoke=True,
                                                      cv=cv,
                                                      n_jobs=1)

    assert isinstance(top_candidates, list)
    assert len(top_candidates) > 0
    # ensure that we didn't sample extensively - with random_search_mult small and small grid, we should have <=3 candidates
    assert len(top_candidates) <= 3
    assert isinstance(best_summary, dict) and 'model' in best_summary and 'score' in best_summary
