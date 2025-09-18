"""
Unit tests for SMOTESampler component.
Usage:
~/proj/shifi/pre_auth.git$ python -m pytest tests/test_smote_sampler.py 

Log:
- 20250924: Initial version, all passing
- 20251004: Added sklearn API compliance
- 20251005: all passing and both smote_sampler.py and test_smote_sampler.py are pylance clean.
"""
import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from imblearn.utils.estimator_checks import parametrize_with_checks
from typing import Callable, cast

# Add src to path for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.components.smote_sampler import MaybeSMOTESampler


# API compliance test stays outside the class
@parametrize_with_checks([MaybeSMOTESampler()])
def test_sklearn_api_compliance(estimator, check):
    check(estimator)

class TestSMOTESampler:
    """Test suite for SMOTESampler component."""
    
    @pytest.fixture
    def binary_sample_data(self):
        """Create sample data for testing."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=200,
            n_features=30,
            n_classes=2,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42
        )
        
        # Convert to DataFrame with some categorical columns
        columns = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)

        # Check for NaN values in the feature columns
        print("NaN values in feature_0:", X_df['feature_0'].isnull().any())
        print("NaN values in feature_1:", X_df['feature_1'].isnull().any())

        # Make last two columns categorical (binary)
        X_df['cat_feature_1'] = (X_df['feature_0'] >= X_df['feature_0'].median()).astype(int)
        X_df['cat_feature_2'] = (X_df['feature_1'] >= X_df['feature_1'].median()).astype(int)

        # Make last two columns categorical (binary)
        median_0 = X_df['feature_0'].median()
        median_1 = X_df['feature_1'].median()
        print("Median of feature_0:", median_0)
        print("Median of feature_1:", median_1)

        for i in range(X_df.shape[0]):
            print(f"{X_df['feature_0'].iloc[i]}, {X_df['feature_1'].iloc[i]}, {X_df['cat_feature_1'].iloc[i]}, {X_df['cat_feature_2'].iloc[i]}, {X_df['cat_feature_1'].dtype}")

        X_df['cat_feature_1'] = (X_df['feature_0'] >= median_0).astype(int)
        X_df['cat_feature_2'] = (X_df['feature_1'] >= median_1).astype(int)

        headers = {
            "feature_cols": columns,
            "categorical_cols": ['cat_feature_1', 'cat_feature_2'],
            "target_cols": ["y"]
        }
        return X_df, y, headers
    
    
    def test_initialization(self, binary_sample_data):
        """Test SMOTESampler initialization."""
        X_df, y, headers = binary_sample_data
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers=headers,
            k_neighbors=3,
            random_state=42
        )
        
        assert sampler.enabled == True
        assert sampler.categorical_feature_names == ['cat_feature_1', 'cat_feature_2']
        assert sampler.k_neighbors == 3
        assert sampler.random_state == 42
    
    def test_disabled_passthrough(self, binary_sample_data):
        """Test that disabled sampler returns original data."""
        X_df, y, headers = binary_sample_data
        
        sampler = MaybeSMOTESampler(enabled=False,
            headers=headers,

            k_neighbors=3,
            random_state=42
        )

        X_res, y_res = sampler.fit_resample(X_df, y)

        pd.testing.assert_frame_equal(cast(pd.DataFrame, X_res), X_df)
        np.testing.assert_array_equal(y_res, y)
    
    def test_enabled_changes_data_shape(self, binary_sample_data):
        """Test that enabled SMOTE changes data shape."""
        X_df, y, headers = binary_sample_data
        original_shape = X_df.shape
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers=headers,
            k_neighbors=3,
            random_state=42
        )
        
        X_res, y_res = sampler.fit_resample(X_df, y)
        X_res = cast(pd.DataFrame, X_res)  # ensure DataFrame
        
        # Should have more samples after SMOTE
        assert X_res.shape[0] > original_shape[0]
        assert X_res.shape[1] == original_shape[1]  # Same number of features
        assert len(y_res) == X_res.shape[0]  # X and y should match
    
    def test_categorical_features_remain_integer(self, binary_sample_data):
        """Test that categorical features remain integers after SMOTE."""
        X_df, y, headers = binary_sample_data
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers=headers,

            k_neighbors=3,
            random_state=42
        )
        
        X_res, y_res = sampler.fit_resample(X_df, y)
        X_res = cast(pd.DataFrame, X_res)

        # Categorical features should be integers
        assert X_res['cat_feature_1'].dtype in ['int64', 'int32']
        assert X_res['cat_feature_2'].dtype in ['int64', 'int32']
        
        # Values should be binary (0 or 1) for one-hot encoded features
        assert set(X_res['cat_feature_1'].unique()).issubset({0, 1})
        assert set(X_res['cat_feature_2'].unique()).issubset({0, 1})
    
    def test_output_is_dataframe(self, binary_sample_data):
        """Test that output is always a DataFrame."""
        X_df, y, headers = binary_sample_data
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers=headers,

            k_neighbors=3,
            random_state=42
        )
        X_res, y_res = sampler.fit_resample(X_df, y)
        
        assert isinstance(X_res, pd.DataFrame)
        assert list(cast(pd.DataFrame, X_res).columns) == list(X_df.columns)
    
    def test_invalid_categorical_features_raises_critical_message(self, caplog, binary_sample_data):
        """Test that invalid categorical feature names are filtered out."""
        X_df, y, headers = binary_sample_data
        
        headers["categorical_cols"] = ['cat_feature_1', 'nonexistent_column']
        
        # Set up logging capture
        caplog.set_level(logging.CRITICAL)
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers= headers,
            k_neighbors=3,
            random_state=42
        )
        with caplog.at_level(logging.CRITICAL), pytest.raises(ValueError, match="nonexistent_column"):
            X_res, y_res = sampler.fit_resample(X_df, y)
            assert "nonexistent_column" in caplog.text
            assert "CRITICAL" in caplog.text
    
    def test_k_neighbors_adjustment(self, binary_sample_data):
        """Test that k_neighbors is adjusted for small minority class."""
        X_df, y, headers = binary_sample_data
        
        # Make sure we have a very small minority class
        minority_size = np.bincount(y).min()
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers=headers,
            k_neighbors=minority_size,  # Too large
            random_state=42
        )
        
        # Should not raise error - k_neighbors should be adjusted
        X_res, y_res = sampler.fit_resample(X_df, y)
        assert isinstance(X_res, pd.DataFrame)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        X_empty = pd.DataFrame()
        y_empty = np.array([])
        headers = {
            "categorical_cols": [],
            "feature_cols": [],
            "target_cols": []
        }
        
        sampler = MaybeSMOTESampler(enabled=True,
            headers=headers,
            k_neighbors=3,
            random_state=42
        )
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            sampler.fit_resample(X_empty, y_empty)
    
    def test_mismatched_lengths_raises_error(self, binary_sample_data):
        """Test that mismatched X and y lengths raise error."""
        X_df, y, headers = binary_sample_data
        
        sampler = MaybeSMOTESampler(headers=headers, enabled=True)
        
        with pytest.raises(ValueError, match="X and y length mismatch"):
            sampler.fit_resample(X_df, y[:-5])  # Remove last 5 elements
    
    def test_get_set_params(self, binary_sample_data):
        """Test parameter getting and setting."""
        X_df, y, headers = binary_sample_data

        sampler = MaybeSMOTESampler(headers=headers, k_neighbors=5, enabled=True)
        
        params = sampler.get_params()
        assert params['k_neighbors'] == 5
        assert params['enabled'] == True
        
        sampler.set_params(k_neighbors=7, enabled=False)
        assert sampler.k_neighbors == 7
        assert sampler.enabled == False
    
    def test_nan_values_in_categorical_columns(self, binary_sample_data):
        """Test handling of NaN values in categorical columns."""
        X_df, y, headers = binary_sample_data
        
        # Introduce some NaN values in categorical column
        X_df.loc[0:5, 'cat_feature_1'] = np.nan
        
        sampler = MaybeSMOTESampler(
            enabled=True,
            headers = headers,
            k_neighbors=3,
            sampling_strategy="",
            random_state=42,
            min_improvement=0.1
        )
    
        with pytest.raises(ValueError, match=r"has \d+ NaN values"):
            X_res, y_res = sampler.fit_resample(X_df, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
