from sklearn.feature_selection import SelectKBest
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE, SMOTENC
from typing import List, Union, Optional

# --------------------------
# Named SMOTE / SMOTENC (name-based categorical mapping)
# --------------------------
class NamedSMOTE(SMOTE):
    """
    SMOTE that accepts categorical_feature_names (list of column names). During fit_resample
    it maps names -> indices relative to the current DataFrame columns.
    """
    def __init__(self, categorical_feature_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.categorical_feature_names = categorical_feature_names or []
        self.categorical_features_indices: List[int] = []

    def fit_resample(self, X, y):
        # Expect X to be a DataFrame (SelectKBestDf ensures this)
        if self.categorical_feature_names:
            if not hasattr(X, "columns"):
                raise TypeError("NamedSMOTE expects a pandas DataFrame (so column names are available).")
            self.categorical_features_indices = []
            for cname in self.categorical_feature_names:
                if cname in X.columns:
                    loc = X.columns.get_loc(cname)
                    if isinstance(loc, (int, np.integer)):
                        self.categorical_features_indices.append(int(loc))
                    elif isinstance(loc, np.ndarray) and loc.dtype == bool:
                        idxs = np.where(loc)[0]
                        if len(idxs) == 1:
                            self.categorical_features_indices.append(int(idxs[0]))
                        else:
                            raise ValueError(f"Column '{cname}' matched multiple columns.")
                    else:
                        raise ValueError(f"Unexpected get_loc return type for '{cname}': {type(loc)}")
                else:
                    logger.debug(f"NamedSMOTE: categorical column '{cname}' not found in current X; skipping.")
            # set numeric indices for SMOTE's internal use
            self.categorical_features = self.categorical_features_indices
        else:
            self.categorical_features = []
        X_res, y_res = super().fit_resample(X, y)
        # convert X_res back to DataFrame if parent returns ndarray
        if isinstance(X_res, np.ndarray) and hasattr(X, "columns"):
            # Parent's SMOTE returns array; rebuild DataFrame with same selected columns
            cols = list(X.columns)
            X_res = pd.DataFrame(X_res, columns=cols)
        return X_res, y_res

class NamedSMOTENC(SMOTENC):
    """
    SMOTENC that accepts categorical_feature_names and maps to indices at fit_resample.
    """
    def __init__(self, categorical_feature_names: List[str], **kwargs):
        if not categorical_feature_names:
            raise ValueError("NamedSMOTENC requires non-empty categorical_feature_names.")
        # initialize with empty categorical_features; set later
        super().__init__(categorical_features=[], **kwargs)
        self.categorical_feature_names = categorical_feature_names
        self.categorical_features_indices: List[int] = []

    def fit_resample(self, X, y):
        if not hasattr(X, "columns"):
            raise TypeError("NamedSMOTENC expects a pandas DataFrame.")
        self.categorical_features_indices = []
        for cname in self.categorical_feature_names:
            if cname not in X.columns:
                logger.debug(f"NamedSMOTENC: categorical '{cname}' not in X; skipping.")
                continue
            loc = X.columns.get_loc(cname)
            if isinstance(loc, (int, np.integer)):
                self.categorical_features_indices.append(int(loc))
            elif isinstance(loc, np.ndarray) and loc.dtype == bool:
                idxs = np.where(loc)[0]
                if len(idxs) == 1:
                    self.categorical_features_indices.append(int(idxs[0]))
                else:
                    raise ValueError(f"Column '{cname}' matches multiple columns.")
            else:
                raise ValueError(f"Unexpected get_loc return type for '{cname}': {type(loc)}")
        self.categorical_features = self.categorical_features_indices
        X_res, y_res = super().fit_resample(X, y)
        if isinstance(X_res, np.ndarray) and hasattr(X, "columns"):
            cols = list(X.columns)
            X_res = pd.DataFrame(X_res, columns=cols)
        return X_res, y_res

def select_smote_strategy(categorical_feature_names: List[str]) -> Union[NamedSMOTE, NamedSMOTENC]:
    """Return NamedSMOTENC if categorical features present, else NamedSMOTE."""
    if categorical_feature_names:
        logger.debug(f"select_smote_strategy -> NamedSMOTENC for {len(categorical_feature_names)} categorical cols.")
        return NamedSMOTENC(categorical_feature_names=categorical_feature_names, random_state=RANDOM_STATE)
    else:
        logger.debug("select_smote_strategy -> NamedSMOTE (no categorical cols).")
        return NamedSMOTE(categorical_feature_names=[], random_state=RANDOM_STATE)

class SelectKBestDf(SelectKBest):
    """
    A custom scikit-learn transformer that handles pandas DataFrames and 
    preserves column names after transformation.

    This class is intended to be used as a wrapper for other scikit-learn 
    transformers that are not DataFrame-aware. It captures feature names 
    during the `fit` step and uses them to reconstruct a DataFrame after 
    the `transform` step.
    Specifically, it wraps SelectKBest so that transform(X: DataFrame) -> DataFrame with selected column names.
    This is important so the SMOTE step receives a DataFrame with column names for NamedSMOTE.
    """
    def fit(self, X, y):
        """
        Fits the transformer to the input data and stores feature names.

        This method first checks if the input `X` is a pandas DataFrame. 
        If it is, it stores the column names in the `feature_names_in_` 
        attribute, which aligns with scikit-learn's convention for storing 
        input feature names. It then passes the underlying NumPy array 
        to the parent class's `fit` method. This ensures compatibility 
        with standard scikit-learn estimators while preserving metadata.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            The training input samples. If a DataFrame, its column names 
            are stored for later use in the `transform` method.
            
        y : array-like, default=None
            The target values. This parameter is ignored but included 
            for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(X, pd.DataFrame):
            # store feature names; call parent on values to avoid potential sklearn warnings
            self.feature_names_in_ = X.columns.to_numpy()
            super().fit(X.values, y)
        else:
            super().fit(X, y)
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        """
        Transforms the input data and returns a DataFrame with preserved column names.

        This method first converts the input `X` to a NumPy array if it is a 
        pandas DataFrame. It then applies the transformation by calling the 
        parent class's `transform` method. If feature names were stored during 
        the `fit` step, it uses the `get_support()` method to identify the 
        columns that were not removed (e.g., in a feature selection step) and 
        uses these names to create a new pandas DataFrame.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            The input data to be transformed.

        Returns
        -------
        transformed : array-like or pandas.DataFrame
            The transformed data. If the original input was a pandas DataFrame 
            and feature names were stored, the output will also be a pandas 
            DataFrame with the correct column names. Otherwise, it returns 
            a standard NumPy array.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        transformed = super().transform(X_arr)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            cols = list(self.feature_names_in_[self.get_support()])
            return pd.DataFrame(transformed, columns=cols, index=(X.index if hasattr(X, "index") else None))
        else:
            return transformed
