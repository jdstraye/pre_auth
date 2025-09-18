import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure `src/` is on sys.path so imports like `components.smote_sampler` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_coordinator import MLPipelineCoordinator
from components.feature_selector import FeatureSelector, FeatureSelectingClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


def make_df(n=12, n_features=3, random_state=42):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(rng.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    return X


def test_validate_pipeline_input_success():
    coord = MLPipelineCoordinator()
    X = make_df(n=12)
    # balanced binary labels with at least 2 per class
    y = np.array([0, 1] * 6)

    report = coord.validate_pipeline_input(X, y)
    assert isinstance(report, dict)
    assert report.get("passed", False) is True


def test_validate_pipeline_input_mismatch_lengths_fails():
    coord = MLPipelineCoordinator()
    X = make_df(n=10)
    y = np.array([0] * 9)  # length mismatch

    report = coord.validate_pipeline_input(X, y)
    assert report.get("passed") is False
    assert any("X and y length mismatch" in str(i) for i in report.get("issues", []))


def test_validate_pipeline_input_min_class_size():
    coord = MLPipelineCoordinator()
    X = make_df(n=10)
    # class 1 has only one sample
    y = np.array([0] * 9 + [1])

    report = coord.validate_pipeline_input(X, y)
    assert report.get("passed") is False
    assert any("Minimum class size" in str(i) for i in report.get("issues", []))


def test_create_pipeline_structure():
    coord = MLPipelineCoordinator()
    base = RandomForestClassifier(random_state=42, n_estimators=5)

    smote_cfg = {}
    fs_cfg = {"max_features": 2, "threshold": None}

    pipeline = coord.create_pipeline(base, smote_cfg, fs_cfg)
    # Should be an imblearn Pipeline and contain the expected step names
    assert hasattr(pipeline, "steps")
    names = [name for name, _ in pipeline.steps]
    assert names == ["smote", "feature_selecting_classifier"]


def test_fit_pipeline_raises_on_invalid_input():
    coord = MLPipelineCoordinator()
    base = RandomForestClassifier(random_state=42, n_estimators=5)
    pipeline = coord.create_pipeline(base, {}, {"max_features": 2})

    # Construct invalid inputs (too-small class size)
    X = make_df(n=10)
    y = np.array([0] * 9 + [1])

    with pytest.raises(ValueError):
        coord.fit_pipeline(pipeline, X, y, validate_input=True)


def test_sklearn_compliance_feature_selector_and_classifier():
    """Run scikit-learn's estimator checks on FeatureSelector and FeatureSelectingClassifier."""
    # FeatureSelector is a transformer
    fs = FeatureSelector()
    check_estimator(fs)

    # FeatureSelectingClassifier wraps an estimator; use a lightweight RF
    fsc = FeatureSelectingClassifier(estimator=RandomForestClassifier(n_estimators=5, random_state=42))
    check_estimator(fsc)


import pytest


@pytest.mark.xfail(reason="Pipeline checks may exercise multioutput y which FeatureSelector does not accept; skip for now")
def test_sklearn_compliance_pipeline_passthrough():
    """Check a scikit-learn Pipeline that uses FeatureSelector as transformer and RF as classifier.
    Marked xfail because some sklearn pipeline checks pass 2D/multioutput y which FeatureSelector does not accept.
    """
    pipeline = Pipeline([
        ("selector", FeatureSelector()),
        ("clf", RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    check_estimator(pipeline)
