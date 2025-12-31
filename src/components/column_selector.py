"""
Simple DataFrame column selector transformer for pipelines.
"""
from __future__ import annotations
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of columns from a pandas DataFrame.

    Parameters
    ----------
    columns : list[str] or None
        Columns to select. If None, no-op.
    """
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns is None or len(self.columns) == 0:
            return X
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                # silently ignore missing columns (occurs when features absent)
                cols = [c for c in self.columns if c in X.columns]
            else:
                cols = list(self.columns)
            return X.loc[:, cols]
        # If X is numpy array, we cannot select by name - return as-is
        return X
