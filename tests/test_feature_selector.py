"""
Unit tests for FeatureSelector component.
Usage:
~/proj/shifi/pre_auth.git$ python -m pytest tests/test_feature_selector.py 

Log:
- 20250924: Initial version, all passing
- 20251003: Failing, newer feature_selector.py. I don't believe it was ever working.
- 20251006: Added SKlearn and imblearn API compliance test parameterized test.

"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, f_classif

# Add src to path for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.components.feature_selector import FeatureSelector, FeatureSelectingClassifier

# API compliance test stays outside the class
@parametrize_with_checks([FeatureSelector()])  # type: ignore
def test_sklearn_api_compliance(estimator, check):
    check(estimator)
    
class TestFeatureSelector:
    """Test suite for FeatureSelector component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        
        columns = [f'feature_{i}' for i in range(20)]
        X_df = pd.DataFrame(X, columns=columns)
        
        return X_df, y
    
    @pytest.fixture
    def base_estimator(self):
        """Create base estimator for feature selection."""
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    def test_scikitlearn_api_compliance(self, base_estimator):
        fs=FeatureSelector(estimator=base_estimator,
            max_features = 1,
            threshold=0.1)
        fs.set_output(transform="default")
        check_estimator(fs)

    def test_initialization(self, base_estimator):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=10,
            threshold=0.1
        )
        
        assert selector.estimator == base_estimator
        assert selector.max_features == 10
        assert selector.threshold == 0.1
    
    def test_fit_with_max_features(self, sample_data, base_estimator):
        """Test fitting with max_features parameter."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=10
        )
        
        selector.fit(X_df, y)
        
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) == 10
        assert all(feat in X_df.columns for feat in selector.selected_features_)
    
    def test_fit_with_threshold(self, sample_data, base_estimator):
        """Test fitting with threshold parameter."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            threshold="median"
        )
        
        selector.fit(X_df, y)
        
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) > 0
        assert len(selector.selected_features_) <= len(X_df.columns)
    
    def test_transform_preserves_dataframe(self, sample_data, base_estimator):
        """Test that transform returns DataFrame with correct columns."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=5
        )
        
        selector.fit(X_df, y)
        X_transformed = selector.transform(X_df)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 5
        assert X_transformed.shape[0] == X_df.shape[0]
        assert list(X_transformed.columns) == list(selector.selected_features_)

    def test_fit_transform(self, sample_data, base_estimator):
        """Test fit_transform method."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=8
        )
        
        X_transformed = selector.fit_transform(X_df, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 8
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) == 8
    
    def test_get_support(self, sample_data, base_estimator):
        """Test get_support method."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=5
        )
        
        selector.fit(X_df, y)
        
        support_mask: np.ndarray = selector.get_support()
        support_indices = selector.get_support(indices=True)
        
        assert len(support_mask) == len(X_df.columns)
        assert support_mask.sum() == 5
        assert len(support_indices) == 5
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=RandomForestClassifier(),
            max_features=5
        )
        
        with pytest.raises(NotFittedError, match="is not fitted yet"):
            selector.transform(X_df)
    
    def test_get_feature_names_out(self, sample_data, base_estimator):
        """Test get_feature_names_out method."""
        X_df, y = sample_data
        
        selector = FeatureSelector(
            estimator=base_estimator,
            max_features=7
        )
        
        selector.fit(X_df, y)
        feature_names = selector.get_feature_names_out()
        
        assert len(feature_names) == 7
        assert all(name in X_df.columns for name in feature_names)


# ============================================================================
# Test FeatureSelector - More Basic Functionality
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


class TestFeatureSelectorBasics:
    """Test basic FeatureSelector functionality."""
    
    def test_initialization(self, estimators):
        """Test FeatureSelector can be initialized."""
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=10
        )
        assert selector.estimator is not None
        assert selector.max_features == 10
        assert selector.threshold is None
    
    def test_fit_model_based(self, simple_classification_data, estimators):
        """Test model-based feature selection."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=10
        )
        
        selector.fit(X, y)
        
        assert hasattr(selector, 'selected_features_')
        assert len(selector.selected_features_) == 10
        assert hasattr(selector, 'feature_names_in_')
        assert len(selector.feature_names_in_) == X.shape[1]
    
    def test_fit_filter_based(self, simple_classification_data):
        """Test filter-based feature selection with chi2."""
        X, y, _, _ = simple_classification_data
        
        # Make data non-negative for chi2
        X = X - X.min().min()
        
        selector = FeatureSelector(
            estimator=None,
            max_features=8,
            score_func=chi2
        )
        
        selector.fit(X, y)
        
        assert hasattr(selector, 'selected_features_')
        assert len(selector.selected_features_) == 8
    
    def test_transform_preserves_dataframe(self, simple_classification_data, estimators):
        """Test that transform returns a DataFrame."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 10
        assert X_transformed.shape[0] == X.shape[0]
        assert all(col in X.columns for col in X_transformed.columns)
    
    def test_fit_transform(self, simple_classification_data, estimators):
        """Test fit_transform method."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        
        X_transformed = selector.fit_transform(X, y)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == 10
    
    def test_column_names_preserved(self, simple_classification_data, estimators):
        """Test that selected column names are preserved."""
        X, y, _, _ = simple_classification_data
        original_columns = X.columns.tolist()
        
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        # All transformed columns should be from original columns
        assert all(col in original_columns for col in X_transformed.columns)
    
    def test_index_preserved(self, simple_classification_data, estimators):
        """Test that DataFrame index is preserved through transform."""
        X, y, _, _ = simple_classification_data
        X.index = [f"sample_{i}" for i in range(len(X))]
        
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert X_transformed.index.equals(X.index)


# ============================================================================
# Test FeatureSelector - Edge Cases
# ============================================================================

class TestFeatureSelectorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_select_all_features(self, simple_classification_data, estimators):
        """Test selecting all features."""
        X, y, n_informative, n_redundant = simple_classification_data
        n_features = X.shape[1]
        
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=n_features
        )

        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert X_transformed.shape[1] == n_features
    
    def test_select_one_feature(self, simple_classification_data, estimators):
        """Test selecting only one feature."""
        X, y, _, _ = simple_classification_data
        
        selector = FeatureSelector(estimator=estimators['rf'], max_features=1)
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert X_transformed.shape[1] == 1
        assert isinstance(X_transformed, pd.DataFrame)
    
    def test_max_features_exceeds_available(self, simple_classification_data, estimators):
        """Test when max_features > available features."""
        X, y, _, _ = simple_classification_data
        n_features = X.shape[1]
        
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=n_features + 10
        )
        with pytest.raises(ValueError, match="max_features ==.*, must be.*"):
            selector.fit(X, y)
            X_transformed = selector.transform(X)
        
    
    def test_threshold_none_max_features_none(self, simple_classification_data, estimators):
        """Test behavior when both threshold and max_features are None."""
        X, y, _, _ = simple_classification_data
        
        selector = FeatureSelector(
            estimator=estimators['rf'],
            max_features=None,
            threshold="median"  # Default
        )
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        # Should select some features based on median threshold
        assert 0 < X_transformed.shape[1] < X.shape[1]
    
    def test_empty_dataframe_raises(self, estimators):
        """Test that empty DataFrame raises appropriate error."""
        X = pd.DataFrame()
        y = np.array([])
        
        selector = FeatureSelector(estimator=estimators['rf'], max_features=5)
        
        with pytest.raises((ValueError, IndexError)):
            selector.fit(X, y)

# ============================================================================
# Test FeatureSelector - Error Handling
# ============================================================================

class TestFeatureSelectorErrors:
    """Test error handling."""
    
    def test_not_fitted_error(self, simple_classification_data, estimators):
        """Test that unfitted selector raises NotFittedError."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        
        with pytest.raises(NotFittedError):
            selector.transform(X)
    
    def test_missing_columns_in_transform(self, simple_classification_data, estimators):
        """Test transform with missing columns."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        selector.fit(X, y)
        
        # Remove a column
        X_missing = X.drop(columns=[X.columns[0]])
        
        # Should raise ValueError in DEBUG_MODE
        from src.utils import gv
        if gv.DEBUG_MODE:
            with pytest.raises(ValueError):
                selector.transform(X_missing)
    
    def test_none_y_raises(self, simple_classification_data, estimators):
        """Test that y=None raises ValueError."""
        X, y, _, _ = simple_classification_data
        selector = FeatureSelector(estimator=estimators['rf'], max_features=10)
        
        with pytest.raises(ValueError, match="y cannot be None"):
            selector.fit(X, None)
    
    def test_score_func_without_max_features(self, simple_classification_data):
        """Test that score_func without max_features raises ValueError."""
        X, y, _, _ = simple_classification_data
        # Ensure non-negative data for chi2
        import sklearn.preprocessing
        X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

        selector = FeatureSelector(
            estimator=None,
            score_func=chi2,
            max_features=None  # Missing required parameter
        )
        
        with pytest.raises(ValueError, match="max_features"):
            selector.fit(X, y)





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
