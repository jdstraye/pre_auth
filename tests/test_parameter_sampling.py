"""
Tests for generate_parameter_samples in eval_utils.py
Usage:
~/proj/shifi/pre_auth.git$ python -m pytest tests/test_parameter_sampling.py 

Log:
- 20250919: Initial version, all passing
"""
import pytest
import numpy as np
from sklearn.model_selection import ParameterSampler
from collections import Counter
from src.eval_utils import generate_parameter_samples
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def param_dist():
    """Example parameter distribution for testing."""
    return {
        "feature_selecting_classifier__max_features": list(range(0, 50)),  # Smaller range for testing
        "feature_selecting_classifier__threshold": np.linspace(0.25, 1, 100).tolist(),
        "other_param": [1, 2, 3],  # Additional parameter for testing
    }

def test_equal_split(param_dist):
    """Test that samples are split equally between max_features and threshold."""
    n_samples = 10
    samples = generate_parameter_samples(param_dist, n_samples, random_state=42)

    # Count samples with max_features and threshold
    maxf_count = sum(1 for s in samples if s["feature_selecting_classifier__max_features"] is not None)
    thresh_count = sum(1 for s in samples if s["feature_selecting_classifier__threshold"] is not None)

    assert len(samples) == n_samples
    assert maxf_count == thresh_count  # Equal split

def test_mutual_exclusivity(param_dist):
    """Test that no sample has both max_features and threshold set."""
    n_samples = 35
    samples = generate_parameter_samples(param_dist, n_samples, random_state=42)
    for i in range(len(samples)):
        print(f"{i}: {samples[i]}")
        maxf = samples[i]["feature_selecting_classifier__max_features"]
        thresh = samples[i]["feature_selecting_classifier__threshold"]
        assert not (maxf is not None and thresh is not None), "Both max_features and threshold are set."
        assert not (maxf is None and thresh is None), "Both max_features and threshold are None."
    assert len(samples) == n_samples

def test_sample_values(param_dist):
    """Test that sample values are drawn from the input distribution."""
    n_samples = 10
    samples = generate_parameter_samples(param_dist, n_samples, random_state=42)
    assert len(samples) == n_samples

    for s in samples:
        maxf = s["feature_selecting_classifier__max_features"]
        thresh = s["feature_selecting_classifier__threshold"]
        other = s["other_param"]

        if maxf is not None:
            assert maxf in param_dist["feature_selecting_classifier__max_features"]
        if thresh is not None:
            assert thresh in param_dist["feature_selecting_classifier__threshold"]
        assert other in param_dist["other_param"]

def test_shuffled_output(param_dist):
    """Test that the output is shuffled (not ordered by max_features/thresh)."""
    n_samples = 10
    samples1 = generate_parameter_samples(param_dist, n_samples, random_state=42)
    samples2 = generate_parameter_samples(param_dist, n_samples, random_state=42)

    # Different seeds should produce different orders
    assert samples1 != samples2

def test_edge_case_empty_distribution():
    """Test behavior with empty or minimal distributions."""
    param_dist = {
        "feature_selecting_classifier__max_features": [1],
        "feature_selecting_classifier__threshold": [0.5],
    }
    n_samples = 2
    samples = generate_parameter_samples(param_dist, n_samples, random_state=42)

    assert len(samples) == n_samples

    # Check that one sample has max_features set and the other has threshold set
    maxf_samples = [s for s in samples if s["feature_selecting_classifier__max_features"] is not None]
    thresh_samples = [s for s in samples if s["feature_selecting_classifier__threshold"] is not None]
    assert len(maxf_samples) == 1
    assert len(thresh_samples) == 1

def test_other_param_distribution(param_dist):
    """Test that other_param values are well-distributed in the final samples."""
    n_samples = 100  # Use a larger sample size for distribution testing
    samples = generate_parameter_samples(param_dist, n_samples, random_state=42)

    # Extract all values of other_param
    other_params = [s["other_param"] for s in samples]

    # Count occurrences of each value
    param_counts = Counter(other_params)

    # Check that all values of other_param appear in the samples
    for val in param_dist["other_param"]:
        assert val in param_counts, f"Value {val} not found in other_param distribution."

    # Optional: Check that the distribution is roughly uniform (adjust tolerance as needed)
    expected_count = n_samples / len(param_dist["other_param"])
    for val, count in param_counts.items():
        assert abs(count - expected_count) <= expected_count * 0.3, (
            f"Value {val} is not well-distributed (expected ~{expected_count}, got {count})."
        )


def test_encoding_smote_compatibility():
    param_dist = {
        'encoding': ['ohe', 'ordinal'],
        'smote__method': ['smotenc', 'smote', 'none'],
        'feature_selecting_classifier__max_features': [10, None],
        'feature_selecting_classifier__threshold': [None, 0.5]
    }
    samples = generate_parameter_samples(param_dist, 50, random_state=42)
    # No sample should have encoding=='ohe' and smote__method=='smotenc'
    for s in samples:
        assert not (s.get('encoding') == 'ohe' and s.get('smote__method') == 'smotenc')