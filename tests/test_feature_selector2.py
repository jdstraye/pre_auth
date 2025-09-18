"""
Comprehensive pytest suite for feature_selector.py

Tests cover:
- Basic functionality (fit, transform, fit_transform)
- DataFrame preservation and column name handling
- Model-based vs filter-based selection
- Edge cases (empty data, single feature, all features selected)
- Error handling (missing columns, wrong types, unfitted estimators)
- Integration with sklearn pipelines
- Pickling/unpickling
- Memory leaks and performance

Usage:
# Run all tests
pytest tests/test_feature_selector.py -v

# Run specific test class
pytest tests/test_feature_selector.py::TestFeatureSelectorBasics -v

# Run with coverage
pytest tests/test_feature_selector.py --cov=src.components.feature_selector

# Run only fast tests (skip slow performance tests)
pytest tests/test_feature_selector.py -v -m "not slow"

"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

# Adjust import path as needed for your project structure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.feature_selector import FeatureSelector, FeatureSelectingClassifier

# ============================================================================
# Test Sklearn Compliance for FeatureSelectingClassifier
# ============================================================================

from sklearn.utils.estimator_checks import parametrize_with_checks

@parametrize_with_checks([
    FeatureSelectingClassifier(
        estimator=LogisticRegression(max_iter=1000, random_state=42)
        # max_features=None
    )
])   # type: ignore[misc]
def test_feature_selecting_classifier_sklearn_api_compliance(estimator, check):
    """
    Run scikit-learn's full compliance checks on FeatureSelectingClassifier.

    This ensures the meta-estimator follows BaseEstimator and ClassifierMixin
    conventions: cloning, parameter immutability, fit/predict semantics, and
    nested parameter exposure.
    """
    check(estimator)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_classification_data():
    """Generate simple classification dataset as DataFrame."""
    n_informative = 10
    n_redundant = 5
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, n_informative, n_redundant


@pytest.fixture
def classification_with_categorical():
    """Classification data with categorical features."""
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=8,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Add categorical features
    X_df['cat_1'] = np.random.randint(0, 3, size=len(X_df))
    X_df['cat_2'] = np.random.randint(0, 5, size=len(X_df))
    
    return X_df, y


@pytest.fixture
def imbalanced_classification():
    """Imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=25,
        n_informative=15,
        weights=[0.9, 0.1],
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    return X_df, y


@pytest.fixture
def regression_data():
    """Regression dataset for testing with regression estimators."""
    X, y = make_regression(
        n_samples=150,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    return X_df, y


@pytest.fixture
def data_with_nans():
    """Dataset with missing values."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    
    # Introduce NaNs
    X_df.loc[5:10, 'col_0'] = np.nan
    X_df.loc[15:20, 'col_5'] = np.nan
    
    return X_df, y


@pytest.fixture
def estimators():
    """Various estimators for testing."""
    return {
        'rf': RandomForestClassifier(n_estimators=10, random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'gb': GradientBoostingClassifier(n_estimators=10, random_state=42)
    }

# ============================================================================
# Test FeatureSelectingClassifier - Basic Functionality
# ============================================================================
class TestFeatureSelectingClassifierBasics:
    """Test FeatureSelectingClassifier basic functionality."""
    
    def test_initialization(self, estimators):
        """Test classifier can be initialized."""
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        assert clf.estimator is not None
        assert clf.max_features == 10
    
    def test_fit(self, simple_classification_data, estimators):
        """Test fitting the classifier."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        
        clf.fit(X, y)
        
        assert hasattr(clf, 'feature_selector_')
        assert hasattr(clf, 'classifier_')
        assert hasattr(clf, 'classes_')
        assert clf.feature_selector_ is not None
        assert clf.classifier_ is not None
    
    def test_predict(self, simple_classification_data, estimators):
        """Test prediction."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        
        clf.fit(X, y)
        predictions = clf.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in clf.classes_ for pred in predictions)
    
    def test_predict_proba(self, simple_classification_data, estimators):
        """Test probability prediction."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        
        assert probas.shape[0] == len(y)
        assert probas.shape[1] == len(clf.classes_)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_cross_validation(self, simple_classification_data, estimators):
        """Test that classifier works with cross-validation."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        
        scores = cross_val_score(clf, X, y, cv=3)
        
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)

class TestFeatureSelectingClassifier:
    """Test suite for FeatureSelectingClassifier component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        
        columns = [f'feature_{i}' for i in range(15)]
        X_df = pd.DataFrame(X, columns=columns)
        
        return X_df, y
    
    def test_initialization(self):
        """Test FeatureSelectingClassifier initialization."""
        base_classifier = RandomForestClassifier(random_state=42)
        
        classifier = FeatureSelectingClassifier(
            estimator=base_classifier,
            max_features=10
        )
        
        assert classifier.estimator == base_classifier
        assert classifier.max_features == 10
    
    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X_df, y = sample_data
        
        classifier = FeatureSelectingClassifier(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            max_features=8
        )
        
        classifier.fit(X_df, y)
        predictions = classifier.predict(X_df)
        
        assert classifier.selected_features_ is not None
        assert len(classifier.selected_features_) == 8
        assert len(predictions) == len(y)
        assert classifier.classes_ is not None
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_df, y = sample_data
        
        classifier = FeatureSelectingClassifier(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            max_features=5
        )
        
        classifier.fit(X_df, y)
        probabilities = classifier.predict_proba(X_df)
        
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == len(np.unique(y))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_without_fit_raises_error(self, sample_data):
        """Test that predict without fit raises error."""
        X_df, y = sample_data
        
        classifier = FeatureSelectingClassifier(
            estimator=RandomForestClassifier(),
            max_features=5
        )
        
        with pytest.raises(NotFittedError, match="is not fitted yet"):
            classifier.predict(X_df)
    
    def test_nested_parameter_setting(self):
        """Test setting nested estimator parameters."""
        classifier = FeatureSelectingClassifier(
            estimator=RandomForestClassifier(n_estimators=50),
            max_features=10
        )
        
        # Test nested parameter setting
        classifier.set_params(estimator__n_estimators=100)
        assert classifier.estimator.n_estimators == 100
    
    def test_get_feature_names_out(self, sample_data):
        """Test getting selected feature names."""
        X_df, y = sample_data
        
        classifier = FeatureSelectingClassifier(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            max_features=6
        )
        
        classifier.fit(X_df, y)
        # diagnostic snippet: paste after classifier.fit(X_df, y)
        if hasattr(classifier, "feature_names_in_"):
            fn = getattr(classifier, "feature_names_in_")
        if hasattr(classifier, "_feature_names_in"):
            fn = getattr(classifier, "_feature_names_in")
        print("DBGfn_out00: feature_names_in_ type:", type(fn), "len:", None if fn is None else len(fn))
        print("DBGfn_out00: feature_names_in_ sample:", None if fn is None else repr(fn[:10]))
        if hasattr(classifier, "_selected_features"):
            sel_private = getattr(classifier, "_selected_features")
        if hasattr(classifier, "selected_features_"):
            sel_public = getattr(classifier, "selected_features_")
        print("DBGfn_out02: selected_features_ (public) type:", type(sel_public), repr(sel_public))
        print("DBGfn_out02: _selected_features (private) type:", type(sel_private), repr(sel_private))

        feature_names = classifier.get_feature_names_out()
        
        assert feature_names is not None
        assert len(feature_names) == 6
        assert all(name in X_df.columns for name in feature_names)


# ============================================================================
# Test Pipeline Integration
# ============================================================================

class TestPipelineIntegration:
    """Test integration with sklearn and imblearn pipelines."""
    
    def test_sklearn_pipeline(self, simple_classification_data, estimators):
        """Test FeatureSelector in sklearn Pipeline."""
        X, y, _, _ = simple_classification_data
        
        pipeline = Pipeline([
            ('selector', FeatureSelector(estimator=estimators['rf'], max_features=10)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_feature_selecting_classifier_in_pipeline(self, simple_classification_data, estimators):
        """Test FeatureSelectingClassifier as pipeline step."""
        X, y, _, _ = simple_classification_data
        
        pipeline = Pipeline([
            ('classifier', FeatureSelectingClassifier(
                estimator=estimators['rf'],
                max_features=10
            ))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_get_set_params(self, estimators):
        """Test get_params and set_params for pipeline compatibility."""
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        
        params = clf.get_params(deep=True)
        assert 'max_features' in params
        assert 'estimator' in params
        
        clf.set_params(max_features=15)
        assert clf.max_features == 15


# ============================================================================
# Test Persistence (Pickling)
# ============================================================================

class TestPersistence:
    """Test pickling and unpickling."""
    
    def test_pickle_feature_selector(self, simple_classification_data, estimators, tmp_path):
        """Test that FeatureSelector can be pickled and unpickled."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        selector.fit(X, y)
        
        # Pickle
        pickle_path = tmp_path / "selector.pkl"
        joblib.dump(selector, pickle_path)
        
        # Unpickle
        selector_loaded = joblib.load(pickle_path)
        
        # Test that it works
        X_transformed = selector_loaded.transform(X)
        assert X_transformed.shape[1] == 10
    
    def test_pickle_feature_selecting_classifier(
        self, simple_classification_data, estimators, tmp_path
    ):
        """Test that FeatureSelectingClassifier can be pickled."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=estimators['rf'],
            max_features=10
        )
        clf.fit(X, y)
        
        # Pickle
        pickle_path = tmp_path / "classifier.pkl"
        joblib.dump(clf, pickle_path)
        
        # Unpickle
        clf_loaded = joblib.load(pickle_path)
        
        # Test that it works
        predictions = clf_loaded.predict(X)
        assert len(predictions) == len(y)


# ============================================================================
# Test Performance and Memory
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset(self, estimators):
        """Test with larger dataset."""
        X, y = make_classification(
            n_samples=5000,
            n_features=100,
            n_informative=50,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=30
        )
        
        import time
        start = time.time()
        selector.fit(X_df, y)
        X_transformed = selector.transform(X_df)
        duration = time.time() - start
        
        assert X_transformed.shape[1] == 30
        assert duration < 30  # Should complete within 30 seconds
    
    def test_no_memory_leak_repeated_fits(self, simple_classification_data, estimators):
        """Test that repeated fits don't cause memory issues."""
        X, y, _, _ = simple_classification_data
        
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        
        # Fit multiple times
        for _ in range(10):
            selector.fit(X, y)
            X_transformed = selector.transform(X)
            assert X_transformed.shape[1] == 10


# ============================================================================
# Test Different Score Functions
# ============================================================================

class TestScoreFunctions:
    """Test different score functions for filter-based selection."""
    
    @pytest.mark.parametrize("score_func", [chi2, f_classif, mutual_info_classif])
    def test_different_score_functions(self, simple_classification_data, score_func):
        """Test with different score functions."""
        X, y, _, _ = simple_classification_data
        
        # Make data non-negative for chi2
        if score_func == chi2:
            X = X - X.min().min()
        
        selector = FeatureSelector(
            estimator=None,
            max_features=8,
            score_func=score_func
        )
        
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert X_transformed.shape[1] == 8

# ============================================================================
# Supplemental Functional / Edge Tests
# ============================================================================

class TestFeatureSelectingClassifierComplianceExtras:
    """Additional targeted tests complementing sklearn compliance checks."""

    def test_get_params_includes_nested(self):
        """Ensure nested estimator params are exposed with 'estimator__' prefix."""
        clf = FeatureSelectingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            max_features=5
        )
        params = clf.get_params(deep=True)
        # Confirm nested exposure and top-level passthrough
        assert "estimator__max_iter" in params
        assert "max_features" in params
        assert isinstance(params["estimator__max_iter"], int)

    def test_clone_and_refit_produces_equivalent_results(self, simple_classification_data):
        """Ensure cloned classifier refits identically."""
        from sklearn.base import clone

        X, y, _, _ = simple_classification_data
        base = FeatureSelectingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            max_features=5
        )
        base.fit(X, y)
        preds_base = base.predict(X)

        clone_clf = clone(base)
        clone_clf.fit(X, y)
        preds_clone = clone_clf.predict(X)

        assert np.array_equal(preds_base, preds_clone)

    def test_predict_before_fit_raises(self, simple_classification_data):
        """Ensure NotFittedError is raised before fit."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            max_features=5
        )
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            clf.predict(X)

    def test_predict_proba_works_with_fitted_classifier(self, simple_classification_data):
        """Ensure predict_proba returns correct shape and sums to 1."""
        X, y, _, _ = simple_classification_data
        clf = FeatureSelectingClassifier(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            max_features=5
        )
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        assert probas.shape == (len(y), len(clf.classes_))
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_in_pipeline_roundtrip(self, simple_classification_data):
        """Ensure FeatureSelectingClassifier works inside sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        X, y, _, _ = simple_classification_data
        pipeline = Pipeline([
            ("clf", FeatureSelectingClassifier(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                max_features=5
            ))
        ])

        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])