import numpy as np
import pandas as pd
from src.components.smote_sampler import MaybeSMOTESampler, NamedSMOTENC


def test_named_smotenc_init_does_not_call_fitresample():
    # Ensure constructing NamedSMOTENC does not attempt to call fit_resample or reference X/y
    # and does not raise UnboundLocalError.
    sm = NamedSMOTENC(feature_names=['a','b'], categorical_feature_names=['nonexistent_cat'])
    assert hasattr(sm, 'feature_names')


def test_maybe_smote_sampler_fallback_when_categorical_missing():
    # Build a small dataset where categorical_feature_names is provided but absent in X
    X = pd.DataFrame({
        'feature_0': [1,2,3,4,5],
        'feature_1': [5,4,3,2,1]
    })
    y = np.array([0,0,0,1,1])

    sampler = MaybeSMOTESampler(enabled=True, headers={'feature_cols': ['feature_0','feature_1'], 'categorical_cols': ['does_not_exist']})
    # Should not raise; fallback to NamedSMOTE or RandomOverSampler is acceptable
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(y_res) >= len(y)
    assert X_res.shape[1] == X.shape[1]
