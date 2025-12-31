import pandas as pd
import numpy as np
from src.components.smote_sampler import MaybeSMOTESampler


def make_tiny_dataset(minority=1):
    # 10 majority, minority variable
    X = pd.DataFrame({f"f{i}":[0]*(10+minority) for i in range(5)})
    y = np.array([0]*10 + [1]*minority)
    return X, y


def test_fallback_on_tiny_minority():
    X, y = make_tiny_dataset(minority=1)
    sampler = MaybeSMOTESampler(enabled=True, k_neighbors=5, allow_fallback=True)
    X_res, y_res = sampler.fit_resample(X, y)
    # fallback should be used because minority=1 < k
    assert sampler.last_fallback_used is True
    assert sampler.last_used_sampler == 'RandomOverSampler'
    # after RandomOverSampler minority should be increased
    unique, counts = np.unique(y_res, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    assert counts_dict.get(1, 0) > 1
