"""
Feature Selection and Classification with DataFrame Support

This module provides two classes for feature selection and classification,
built to be highly compatible with pandas DataFrames and scikit-learn pipelines.
It leverages SelectFromModel for model-based feature selection and SelectKBest
for filter-based selection, ensuring data integrity checks using the DataValidator
in debug_library.py.

Classes:
- FeatureSelector: A DataFrame-aware transformer inheriting from SelectFromModel,
  capable of model-based or filter-based selection.
- FeatureSelectingClassifier: A classifier wrapper that manages the feature
  selection process internally before training the final estimator.  It uses 
  `FeatureSelector` internally to ensure the same features are used for feature selection,
  training, and prediction, maintaining consistency.

Key Features:
- **DataFrame Support**: Both classes handle pandas DataFrames natively, preserving
  column names and indices.
- **Validation**: Uses `DataValidator` in debug_library to check for data corruption,
  missing values, and other issues.
- **Flexibility**: Supports model-based selection (estimator/threshold) and
  filter-based selection (score_func/max_features).

Log
-----
- [20251019] Merged all the FeatureSelector tests from tests/test_feature_selector.py and tests/test_feature_selector2.py
        - 61 passed, 132 warnings in 4.02s
        - None of those 132 warnings indicate an actual problem. They are normal
        side-effects of scikit-learn compliance tests and can safely be ignored or
        filtered out in the test suite.
- [20251109] Back at it. Somehow, FeatureSelectorBase has problems and nothing works.
             After fixing, 61 passed, 132 warnings in 3.54s in tests/test_feature_selector.py
- [20251109] Initial run with tests/test_feature_selector2.py: 43 failed, 27 passed, 18 warnings in 9.25s
"""

import sys
import pandas as pd
import numpy as np
import logging
from scipy import sparse
from scipy.sparse import issparse, spmatrix, csc_matrix, csr_matrix
from pandas import RangeIndex
from typing import Optional, Union, List, Dict, Any, Tuple, Callable, cast
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# NOTE: Assumes src.debug_library is correctly set up in the environment
from src.debug_library import debug_pipeline_step, DataValidator
from src.utils import setup_logging, gv, debug_setup_logging, get_logger

# Standard logging configuration for the module
setup_logging(gv.LOG_DIR/"feature_selector.log")
#debug_setup_logging()

class FeatureSelectorBase ():
    """
    Base class for feature selection transformers.

    This class provides shared methods but expects subclasses to define:
    - estimator
    - max_features  
    - threshold
    - prefit
    - score_func
    It handles input validation, metadata recording, and logging.
    """
    
    # Declare that subclasses must have these attributes
    # These are just for type checking - actual values set in subclass __init__
    estimator: Optional[BaseEstimator]
    max_features: Optional[int]
    threshold: Optional[Union[str, float]]
    prefit: bool
    score_func: Optional[Callable]

    def _validate_and_record_input(self,
                                   X: Union[pd.DataFrame, np.ndarray, pd.Series, spmatrix],
                                   y: Optional[Union[pd.DataFrame, np.ndarray]],
                                   require_y: bool = False,
                                   record_metadata: bool = True,
                                   feature_names: Optional[list[np.ndarray]] = None) -> Tuple[Union[np.ndarray, spmatrix], Optional[Union[np.ndarray, spmatrix]]]:
        """
        Validate inputs and record metadata for format preservation.
        

        Records the input type (DataFrame, array, sparse, Series) and associated
        metadata (column names, index, dtypes) to enable returning output in the
        same format during transform().
        

        Parameters
        ----------
        X : DataFrame, array, or sparse matrix
            Input features. Must be 2D (n_samples, n_features) for numpy arrays.
        y : DataFrame, Series, or array, optional
            Target values.
        require_y : bool, default=False
            Whether y is required (e.g., True for fit() when using SelectFromModel
            but False for transform() where only X is modified).
        feature_names : list of str, optional
            Feature names if X is an array.
        record_metadata : bool, default=True
            Whether to record metadata about the input features. Metadata is used
            to restore the output format in transform() and should only be True for 
            fit() because transform() should not modify the input features (cannot 
            modify __dict__ in SKLearn compliance terminology).

        Returns
        -------
        X_validated : array or sparse matrix
            Validated feature matrix.
        y_validated : array or sparse matrix
            Validated target array.

        Raises
        ------
        TypeError
            If X or y are not supported types.

        Notes
        -----
        There used to be a golden_df attribute that stored the original DataFrame and all 
        the metadata that could be derived from it. It is still here but not used. It has 
        been commented out because it caused multiple problems with SKLearn compliance and
        memory usage.
        """
        lg = self._get_logger()

        # Record input type and columns
        # Record input type
        if record_metadata:
            self._X_is_dataframe = isinstance(X, pd.DataFrame)
            self._X_is_ndarray = isinstance(X, np.ndarray)
            self._X_is_pandas_series = isinstance(X, pd.Series)
            self._X_is_sparse = sparse.issparse(X)
            self._X_is_catchall = not (self._X_is_dataframe or self._X_is_pandas_series or self._X_is_sparse or self._X_is_ndarray)
            self._y_is_dataframe = isinstance(y, pd.DataFrame)
            self._y_is_ndarray = isinstance(y, np.ndarray)
            self._y_is_pandas_series = isinstance(y, pd.Series)
            self._y_is_sparse = sparse.issparse(y)
            self._y_is_catchall = not (self._y_is_dataframe or self._y_is_pandas_series or self._y_is_sparse or self._y_is_ndarray)
#20251014-deprecated             #self.golden_df = pd.DataFrame()
            self._feature_names_in = None
        X_recorded = None
        y_recorded = None

        # Convert and validate X
        if self._X_is_dataframe:
            # Safely, capture DF metadata
            # This should be self.feature_names_in_ after fit(). Setup X as well.
            X_df = cast(pd.DataFrame, X)
            if record_metadata:
                self._feature_names_in = np.array([str(c) for c in list(getattr(X_df, 'columns', []))], dtype=object)
                self._X_index = getattr(X_df, 'index', None)
                self._X_dtypes = getattr(X_df, 'dtypes', None)
                self._validator.validate_frame(X_df, "feature_selector_fit_input")
#20251014-deprecated                 self.golden_df = X_df.copy(deep=True)
            X_recorded = X_df.values
        elif self._X_is_pandas_series:
            X_pds = cast(pd.Series, X)
            X_recorded = X_pds.to_numpy().reshape(-1, 1)  # Ensure 2D for Series
            if record_metadata:
                self._X_index = getattr(X_pds, 'index', None)
                self._X_dtype = getattr(X_pds, 'dtype', None)
                self._feature_names_in = np.array([X_pds.name if X_pds.name is not None else "feature0"])
#20251014-deprecated                 self.golden_df = pd.DataFrame(X_recorded, columns=[X_pds.name], index=self._X_index)    
        elif self._X_is_ndarray:
            # Create metadata to the extent possible
            X_nda = cast(np.ndarray, X)
            X_recorded = X_nda
#20251014-deprecated             if record_metadata:
#20251014-deprecated                 self.golden_df = pd.DataFrame(
#20251014-deprecated                     X_recorded,
#20251014-deprecated                     columns=self._feature_names_in,
#20251014-deprecated                     index=self._X_index
#20251014-deprecated                 )
        else:
            # Handle ndarray and any other array-like object
            try:
                X_recorded = np.asarray(X)  # Converts array-like objects
#20251014-deprecated                 if record_metadata:
#20251014-deprecated                     self.golden_df = pd.DataFrame(
#20251014-deprecated                         X_recorded,
#20251014-deprecated                         columns=self._feature_names_in,
#20251014-deprecated                         index=self._X_index
#20251014-deprecated                     )
                lg.warning(f"Input X should be a pandas DataFrame or numpy ndarray, not {type(X) = }.")
            except Exception as e:
                raise TypeError(f"Input X should be a pandas DataFrame or numpy ndarray, not {type(X) = }.\n Error: {e}")

        # Validate X early to catch 1D inputs
        X_validated = check_array(
            X_recorded,
            accept_sparse=True,
            dtype=None,
            force_all_finite=False,
            ensure_2d=True,
            allow_nd=False
        )

        # Set feature names for numpy arrays after validation
        if (self._X_is_ndarray or self._X_is_catchall) and record_metadata:
            if feature_names:
                self._feature_names_in = np.array(feature_names)
            else:
                self._feature_names_in = np.array([f"feature{i}" for i in range(X_validated.shape[1])])
                self._X_index = pd.RangeIndex(start=0, stop=X_validated.shape[0], step=1)

        # Handle y
        if y is None:
            if require_y and self.score_func is not None:
                raise ValueError("y cannot be None when using a score_func for supervised feature selection.")
            if require_y and self.estimator is not None and not self.prefit:
                raise ValueError("y cannot be None when fitting an estimator for feature-based selection.")
            # If unsupervised selection (e.g., prefit model without y), allow y to be None
            y_recorded = None #np.array([])  # Dummy placeholder

        elif self._y_is_dataframe:
            # Safely, capture DF metadata
            # This should be self.feature_names_in_ after fit(). Setup X as well.
            y_df = cast(pd.DataFrame, y)
            if record_metadata:
                self._feature_names_in = getattr(y_df, 'columns', None)
                self._y_index = getattr(y_df, 'index', None)
                self._y_dtypes = getattr(y_df, 'dtypes', None)
                self._validator.validate_frame(y_df, "feature_selector_fit_y_input")
            y_recorded = y_df.values
        elif isinstance(y, list):
            y_recorded = np.array(y)
        elif self._y_is_pandas_series:
            y_pds = cast(pd.Series, y)
            if record_metadata:
                self._y_index = getattr(y_pds, 'index', None)
                self._y_dtypes = getattr(y_pds, 'dtype', None)
            y_recorded = y_pds.to_numpy(self._y_dtypes)
        elif self._y_is_ndarray:
            y_recorded = cast(np.ndarray, y)
            if record_metadata:
                self._y_index = pd.RangeIndex(y.shape[0])
                self._y_dtypes = y_recorded.dtype
        else:
            # Handle ndarray, list, and any array-like object
            try:
                y_recorded = np.asarray(y)  # Converts array-like objects
            except Exception as e:
                raise TypeError(f"Input y should be a pandas DataFrame, Series, numpy ndarray, or list, not {type(y) = }.\n Error: {e}")

        if y_recorded is not None:
            y_validated = check_array(
                y_recorded,
                accept_sparse=True,
                dtype=None,  # Keep original dtype
                force_all_finite=False,  # Check for NaN/inf
                ensure_2d=False,
                allow_nd=False
            )

#20251014-deprecated             # Add target to golden_df for reference
#20251014-deprecated             if record_metadata:
#20251014-deprecated                 ## Rearrange y according to _X_index instead of _y_index.
#20251014-deprecated                 y_validated_df = pd.DataFrame(
#20251014-deprecated                     y_validated,
#20251014-deprecated                     columns=["target"],
#20251014-deprecated                     index=self._X_index  # Force the index to match _X_index
#20251014-deprecated                 )
#20251014-deprecated 
#20251014-deprecated                 ## Verify indices match
#20251014-deprecated                 if not self.golden_df.index.equals(y_validated_df.index):
#20251014-deprecated                     raise ValueError("Index mismatch between X and y after validation."
#20251014-deprecated                                       "golden_df index:", self.golden_df.index,
#20251014-deprecated                                       "y_validated_df index:", y_validated_df.index
#20251014-deprecated                     )
#20251014-deprecated                 
#20251014-deprecated                 ## Concatenate the DFs
#20251014-deprecated                 self.golden_df = pd.concat(
#20251014-deprecated                     [self.golden_df,
#20251014-deprecated                      y_validated_df],
#20251014-deprecated                     axis=1)

        else:
            y_validated = None

        if self._feature_names_in is None:
            lg.debug(f"Could not determine feature names from input X of type {type(X)}\n {X = }.")
            raise ValueError("Feature names could not be determined from input X.")
        return X_validated, y_validated

    def _get_logger(self):
        """
        Return a logger by name. Resolve it at call time so we do NOT store the
        Logger object on the instance (keeps the instance pickleable).
        Get logger instance without storing it (maintains picklability).
        
        Returns
        -------
        logging.Logger
            Logger configured for this class.
        """
        if not hasattr(self, "_logger_name") or self._logger_name is None:
            self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        name = getattr(self, "_logger_name", f"{__name__}.{self.__class__.__name__}.PicklableFeatureSelector")
        lg = get_logger(name)
        return lg

    def _set_fittedattributes_and_privatevars(self):
        """
        Set fitted attributes and private variables after fitting.

        This method sets the standard fitted attributes required by scikit-learn
        as well as private variables used internally for tracking selected features.

        Fitted attributes set:
        - feature_names_in_
        - n_features_in_
        - support_
        - selected_features_

        Private variables set:
        - _feature_names_in
        - _n_features_in
        - _support
        - _selected_features
        """
        lg = self._get_logger()  # ADD THIS LINE

        # Compute effective parameters locally first (no mutation yet)
        use_defaults = self.score_func is None and self.estimator is None and self.max_features is None
        effective_estimator = LogisticRegression(max_iter=1000, random_state=42) if use_defaults else self.estimator
        effective_score_func = None if use_defaults else self.score_func
        effective_max_features = None if use_defaults else self.max_features
        effective_threshold = -np.inf if use_defaults else self.threshold
        
        # Key fix: only use "median" threshold if max_features is NOT specified
        # When max_features IS specified, threshold should be None so SelectFromModel uses max_features
        if self.max_features is not None:
            effective_threshold = -np.inf  # Let max_features control selection; -np.inf will force to use max_features instead of threshold.
        else:
            effective_threshold = "mean" if use_defaults and self.threshold is None else self.threshold

        effective_prefit = self.prefit

        # Store as private attributes (leading _ convention)
        self._internal_estimator = effective_estimator
        self._internal_score_func = effective_score_func
        self._internal_max_features = effective_max_features
        self._internal_threshold = effective_threshold
        self._internal_prefit = effective_prefit

        lg.debug(f"DBGSEL00: {use_defaults = } = {self.score_func = } * {self.estimator = } * {self.max_features = }\n"
                     f"\n {effective_threshold = }, {self.max_features = } & {use_defaults = } * {self.threshold = }"
                     f"\n {self._internal_estimator = }"
                     f"\n {self._internal_score_func = }"
                     f"\n {self._internal_max_features = }"
                     f"\n {self._internal_threshold = }"
                     f"\n {self._internal_prefit = }"
                 )

        ## Data validation setup
        self._validator = DataValidator("FeatureSelector")

        ## Pickleable logger
        self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

        ## Internal state, not parameters
        self.feature_names_in_: np.ndarray # SKLearn's attribute
        self._feature_names_in: Optional[np.ndarray] # Private attribute of feature_names_in_
        self.selected_features_: List[str] # SKLearn's attribute - An array selecting the features that were selected, [True, False, True, etc.] or [0 2 5 ...]
        self._selected_features: Optional[List[str]] # Private attribute of selected_features_
        self.support_: np.ndarray # SKLearn's attribute - An array selecting the features that were selected, [True, False, True, etc.] or [0 2 5 ...]
        self._support: Optional[np.ndarray] # Private attribute of support_
        self.n_features_in_: int # SKLearn's attribute - The number of features seen during fit
        self._n_features_in: Optional[int] # Private attribute of n_features_in_
        self._X_is_catchall: bool = False
        self._X_is_dataframe: bool = False
        self._X_is_ndarray: bool = False
        self._X_is_pandas_series: bool = False
        self._X_is_sparse: bool = False
        self._y_is_catchall: bool = False
        self._y_is_dataframe: bool = False
        self._y_is_ndarray: bool = False
        self._y_is_pandas_series: bool = False
        self._y_is_sparse: bool = False
        self._internal_selector: Optional[Union[SelectFromModel, SelectKBest]] = None
        self._X_index: Optional[pd.Index] = None
        self._X_dtypes: Optional[pd.Series] = None
        self._X_dtype: Optional[np.dtype] = None
        self._y_index: Optional[pd.Index] = None
        self._y_dtype: Optional[np.dtype] = None
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : list of str, optional
            Not used. Present for sklearn API compatibility.
        
        Returns
        -------
        feature_names_out : ndarray of str
            Names of features that will be output by transform(). 
            This would be the private attribute, _selected_features, which should be
            `feature_names_in_[support_]`.
        
        Raises
        ------
        NotFittedError
            If called before fit.
        """
        check_is_fitted(self, ["_selected_features"])

        try:
            if self.selected_features_ is not None:
                # Force plain str to avoid type mismatches (np.str_ vs str)
                names = [str(name) for name in self.selected_features_]
                return np.array(names, dtype=object)
            if self._selected_features is not None:
                names = [str(name) for name in self._selected_features]
                return np.array(names, dtype=object)
            return np.array([], dtype=object)
        except Exception as e:
            lg = self._get_logger()
            lg.error(f"Error in get_feature_names_out: {e}")
            raise ValueError(f"Could not get feature names: {e}")

class FeatureSelector(FeatureSelectorBase, BaseEstimator, TransformerMixin):
    """
    DataFrame-aware feature selector supporting both model-based and filter-based selection.

    This transformer performs feature selection while preserving pandas DataFrame
    metadata, such as column names and index. It delegates to SelectFromModel or SelectKBest based on the
    presence of `score_func`.

    Parameters
    ----------
    estimator : estimator object, optional
        The base estimator from which to select features (used only if `score_func`
        is None). Must have `feature_importances_` or `coef_` attribute.
    
    threshold : str or float, default='median'
        The threshold value for model-based selection (used only if `score_func`
        is None).
    
    max_features : int, optional
        The maximum number of features to select (k). **Required if `score_func`
        is specified.** Takes precedence over `threshold` when used with a model.
        
    prefit : bool, default=False
        Whether a prefit estimator is passed.

    score_func : callable, optional
        A scoring function (e.g., `chi2`). If provided, `SelectKBest` is used.

    Attributes
    ----------
    feature_names_in_ : ndarray of str
        Names of features seen during :term:`fit`.

    n_features_in_ : int
        Number of features seen during fit.
    
    _selected_features : ndarray of shape (n_features_selected_,)
        Names of features that were selected after fit().
    
    support_ : ndarray of shape (n_features_in_,), dtype=bool
        Boolean mask indicating which features were selected. True indicates
        the feature at that index was selected.
    
    _internal_selector : SelectFromModel or SelectKBest
        The fitted selector instance used internally for transformations.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> import pandas as pd
    >>> 
    >>> # Create sample data
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    >>> 
    >>> # Model-based selection
    >>> selector = FeatureSelector(
    ...     estimator=RandomForestClassifier(n_estimators=10, random_state=42),
    ...     max_features=5,
    ...     score_func=None
    ... )
    >>> X_selected = selector.fit_transform(X_df, y)
    >>> print(f"Selected {X_selected.shape[1]} features")
    >>> print(f"Selected features: {selector.selected_features_}")
    >>>
    >>> # Filter-based selection
    >>> from sklearn.feature_selection import chi2
    >>> selector = FeatureSelector(score_func=chi2, max_features=5)
    >>> X_selected = selector.fit_transform(X_df, y)
    >>>
    >>> # Works with numpy arrays too
    >>> X_arr = X_df.values
    >>> X_selected = selector.fit_transform(X_arr, y)  # Returns numpy array

    Notes
    -----
    - Input format is automatically preserved: DataFrame → DataFrame, array → array,
      sparse → sparse
    - When using `score_func`, `max_features` is required
    - When using `estimator`, either `max_features` or `threshold` can be used
    - For sparse matrices, avoids unnecessary densification
    - Column names are preserved for DataFrame inputs
    
    See Also
    --------
    sklearn.feature_selection.SelectFromModel : Model-based feature selection
    sklearn.feature_selection.SelectKBest : Filter-based feature selection
    sklearn.feature_selection.chi2 : Chi-squared statistic for feature selection
    sklearn.feature_selection.f_classif : ANOVA F-value for feature selection    """

    # Type annotations
    ''' 
    Fitted attributes (those ending in _) must not be set in __init__.
    They are assigned during fit, and their existence signals that the estimator
    is fitted. I am also creating private versions for myself that will be used 
    for checking.
    '''
    def __init__(self,
                 estimator: Optional[BaseEstimator] = None,
                 max_features: Optional[int] = None,
                 threshold: Optional[Union[str, float]] = None,
                 prefit: bool = False,
                 score_func: Optional[Callable] = None
                 ):
        """
        Initialize the FeatureSelector.
        Parameters
        ----------
        estimator : estimator object, optional
            The base estimator from which to select features (used only if `score_func`
            is None). Must have `feature_importances_` or `coef_` attribute.
        threshold : str or float, default='median'
            The threshold value for model-based selection (used only if `score_func`
            is None). Must be in the form of a float or a string that can be parsed
            as a float (e.g., 'median', 'mean').
        max_features : int, optional
            The maximum number of features to select (used only if `score_func` is
            None). If None, all features are selected.
        prefit : bool, default=False
            Whether a prefit estimator is passed.
        score_func : callable, optional
            A scoring function (e.g., `chi2`). If provided, `SelectKBest` is used.
        
            NOTES
            -----
            Per SKLearn convention, these attributes must not be permutated. That means
            they must be set in __init__ exactly as they were passed. Any modifications,
            such as for default values, are done in fit(). For logic that used to be here,
            see the # compute_defaults comment section in fit().
            """
        # Set attributes on the instance (always for SKLearn compliance)
        self.estimator = estimator
        self.max_features = max_features
        self.threshold = threshold
        self.prefit = prefit
        self.score_func = score_func
        
    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and contained subobjects.
        
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params['score_func'] = self.score_func
        return params

    def _check_fit_quality(self):
        """
        Check if the estimator is fitted and that all fitted attributes are in good order
        
        Raises
        ------
        ValueError or TypeError
            Highlighting inappropriate values/types if the estimator were well-fitted.
        """
        lg = self._get_logger()

        # Check the quality of the fit
        try:
            ## Check existence of fitted attributes
            check_is_fitted(self, [
                'n_features_in_',
                'feature_names_in_',
                'support_',
                '_support',
                '_internal_selector',
            ])

            ## Further checks on types and values of fitted attributes
            if not isinstance(self.n_features_in_, int):
                raise TypeError(f"BAD FIT- n_features_in_ must be int, got {type(self.n_features_in_)}.")
            if self.n_features_in_ < 1:
                raise ValueError(f"BAD FIT- n_features_in_ must be >= 1, got {self.n_features_in_}.")
            if not isinstance(self.feature_names_in_, np.ndarray):
                dtype_str = f" with dtype {self.feature_names_in_.dtype}" if isinstance(self.feature_names_in_, np.ndarray) else ""
                raise TypeError(f"BAD FIT- feature_names_in_ must be a numpy array; got {type(self.feature_names_in_)}{dtype_str}.")
            if self.feature_names_in_.shape[0] != self.n_features_in_:
                raise ValueError(
                    f"BAD FIT- feature_names_in_ length {self.feature_names_in_.shape[0]} != n_features_in_ ({self.n_features_in_})."
                )
            if self._support is None:
                raise ValueError("BAD FIT- FeatureSelector is not fitted: _support is None.")
            if np.any(pd.isna(self._support)):
                raise ValueError("BAD FIT- FeatureSelector._support contains None or NaN values.")
            if not isinstance(self._support, np.ndarray):
                raise TypeError(f"BAD FIT- _support must be a numpy array, got {type(self._support)}.")
            if self._support.dtype != bool:
                raise TypeError(f"BAD FIT- _support must be boolean mask, got dtype={self._support.dtype}.")
            if self._support.shape[0] != self.n_features_in_:
                raise ValueError(
                    f"BAD FIT- Support mask length {self._support.shape[0]} != n_features_in_ ({self.n_features_in_})."
                )
            if self.support_ is None:
                raise ValueError("BAD FIT- FeatureSelector is not fitted: support_ is None.")
            if np.any(pd.isna(self.support_)):
                raise ValueError("BAD FIT- FeatureSelector.support_ contains None or NaN values.")
            if not isinstance(self.support_, np.ndarray):
                raise TypeError(f"BAD FIT- support_ must be a numpy array, got {type(self.support_)}.")
            if self.support_.dtype != bool:
                raise TypeError(f"BAD FIT- support_ must be boolean mask, got dtype={self.support_.dtype}.")
            if self.support_.shape[0] != self.n_features_in_:
                raise ValueError(
                    f"BAD FIT- Support mask length {self.support_.shape[0]} != n_features_in_ ({self.n_features_in_})."
                )
        except Exception as e:
            lg.critical(f"Estimator not fitted: {e}")
            raise

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, pd.Series, spmatrix],
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureSelector":
        """
        Public fit: manages recording incoming type (DF, Series, array) as well as data dtype,
        converting incoming type to NumPy arrays, passing the data to internal _fit, and 
        converting the _fit output back to the incoming dtype.

        In the flow, fit should take X and figure out which features of X to keep. 
        Therefore, it should return itself with a few more attributes set so that the correct
        features can be taken in transform(). The attributes to set are:
        SK/IMBLearn required:
        - feature_names_in_
        - n_features_in_
        - support_
        Private:
        - _feature_names_in
        - _n_features_in
        - _support
        - _selected_features
        
        For sklearn compliance, the __init__ method does nothing except store the class
        parameters. All dealings with defaults has to be done in fit(). There are also strict
        naming problems. Therefore, fit() has duplicates of the parameters for internal attributes.
        They are usually name self._internal_<attr>, such as self._internal_estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Training data. Can be:
            - pandas DataFrame (column names preserved)
            - numpy array
            - scipy sparse matrix

        y : array-like of shape (n_samples,), optional
            Target values. Required for supervised feature selection.

        feature_names : list of str, optional
            Feature names to use if X is an array without column names.
            Ignored if X is a DataFrame.

        Returns
        -------
        self : FeatureSelector
            Fitted selector with the following attributes set:
            - feature_names_in_
            - n_features_in_
            - selected_features_
            - support_

        Raises
        ------
        ValueError
            If score_func is provided but max_features is None.
            If y is None but required for the selection method.
        TypeError
            If X is not DataFrame, array, or sparse matrix.
        """
        self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        lg = self._get_logger()

        self._set_fittedattributes_and_privatevars()

        # Validate and record input
        X_validated, y_validated = self._validate_and_record_input(
            X, 
            y, 
            require_y=True, 
            record_metadata=True,
            feature_names=None
        )

        # Core fit
        self._fit(X_validated, y_validated)

        # Check that _fit() set fitted attributes correctly.
        if (hasattr(self, "feature_names_in_") and 
            self._feature_names_in is not None and 
            not np.array_equal(self.feature_names_in_, self._feature_names_in)):
                raise ValueError (f"'feature_names_in_' set prior to '{__name__}.{__class__}' and X column names do not match."
                                  f"\n {self.feature_names_in_.tolist() = }"
                                  f"\n {self._feature_names_in.tolist() = }"
                                  )
        # If _fit() did not set the required attributes, do it here.
        ## feature_names_in_ and n_features_in_ = len(feature_names_in_)
        if not hasattr(self, "feature_names_in_") or getattr(self, "feature_names_in_") is None:
            setattr(self, "feature_names_in_", self._feature_names_in)
            setattr(self, "n_features_in_", len(self.feature_names_in_))

        ## support_
        if not hasattr(self, "support_") or getattr(self, "support_") is None:
            setattr(self, "support_", self._support)

        ## selected_features_
        if not hasattr(self, "selected_features_") or getattr(self, "selected_features_") is None:
            setattr(self, "selected_features_", self._selected_features)

        self._check_fit_quality()

        return self

    def _generate_output(self, X : Union[np.ndarray, spmatrix], index: Optional[pd.Index] = None) -> Union[pd.DataFrame, pd.Series, np.ndarray, spmatrix]:
        """
        Convert transformed data back to original input format.
        
        Restores the output to match the input type that was provided to fit(),
        preserving metadata like column names, index, and dtypes where applicable.
        
        Parameters
        ----------
        X : array or sparse matrix
            Transformed data with selected features.
        
        Returns
        -------
        output : DataFrame, Series, array, or sparse matrix
            Data in original input format with appropriate metadata restored.
        
        Raises
        ------
        ValueError
            If Series output requested but multiple features were selected.
        TypeError
            If input type is unrecognized.
        """
        lg = self._get_logger()

        if issparse(X):
            # Determine index to use for output construction; prefer provided index argument
            # Choose a valid index object: prefer provided 'index' if it's sequence-like
            candidate = index if index is not None else getattr(self, '_X_index', None)
            if candidate is not None and hasattr(candidate, '__len__') and not callable(candidate) and len(candidate) == X.shape[0]:
                out_index = candidate
            else:
                out_index = pd.RangeIndex(start=0, stop=X.shape[0], step=1)
            if self._X_is_dataframe:
                # Create DataFrame with selected feature names and index
                ## Convert X to dense if currently sparse
                X_dense = cast(csr_matrix, X).toarray()
                X_df = pd.DataFrame(X_dense, columns=self._selected_features, index=out_index)
                # Restore dtypes where possible
                if hasattr(self, '_X_dtypes') and self._X_dtypes is not None:
                    for col in X_df.columns:
                        if col in self._X_dtypes.index:
                            try:
                                X_df[col] = X_df[col].astype(self._X_dtypes[col])
                            except (ValueError, TypeError) as e:
                                lg.warning(f"Could not restore dtype for column {col}: {e}")
                return X_df
            else: # Sparse input but not DataFrame - return as is
                return X
        if self._X_is_ndarray:
            return np.asarray(X)
        if self._X_is_pandas_series:
            series_kwargs = {}
            if X.shape[1] != 1:
                raise ValueError("Cannot return Series when multiple features are selected.")
            series_kwargs["dtype"] = np.dtype(self._X_dtype) if self._X_dtype is not None else None
            # Prefer the explicit provided index (from transform), else use recorded index
            candidate = index if index is not None else (getattr(self, '_X_index', None) if (hasattr(self, '_X_index') and self._X_index is not None) else None)
            if candidate is not None and hasattr(candidate, '__len__') and not callable(candidate) and len(candidate) == X.shape[0]:
                series_kwargs["index"] = candidate
            else:
                series_kwargs["index"] = pd.RangeIndex(start=0, stop=X.shape[0], step=1)
            series_kwargs["name"] = self._selected_features[0] if self._selected_features else None
            return pd.Series(**series_kwargs)
        elif self._X_is_catchall: # This should catch anything else (i.e., the "else" statement below)
            return np.asarray(X)
        if self._X_is_dataframe:
            # Create DataFrame with selected feature names and index
            ## Case of sparse X is addressed above, so here X is dense
            X_dense = cast(np.ndarray, X)
            candidate = index if index is not None else getattr(self, '_X_index', None)
            if candidate is not None and hasattr(candidate, '__len__') and not callable(candidate) and len(candidate) == X.shape[0]:
                out_index = candidate
            else:
                out_index = pd.RangeIndex(start=0, stop=X.shape[0], step=1)
            X_df = pd.DataFrame(X_dense, columns=self._selected_features, index=out_index)
            # Restore dtypes where possible
            if hasattr(self, '_X_dtypes') and self._X_dtypes is not None:
                for col in X_df.columns:
                    if col in self._X_dtypes.index:
                        try:
                            X_df[col] = X_df[col].astype(self._X_dtypes[col])
                        except (ValueError, TypeError) as e:
                            lg.warning(f"Could not restore dtype for column {col}: {e}")
            return X_df

        # Fallback for unrecognized input type
        lg.critical(f"Unrecognized input type flags: "
                    f"\n {self._X_is_dataframe = }, "
                    f"\n {self._X_is_ndarray = }, "
                    f"\n {self._X_is_pandas_series = }, "
                    f"\n {self._X_is_sparse = }. Returning numpy array.")
        return np.asarray(X)
    
    @debug_pipeline_step("FeatureSelector_Fit")
    def _fit(self, 
             X: Union[np.ndarray,spmatrix], 
             y: Optional[Union[np.ndarray,spmatrix]]) -> 'FeatureSelector':
        """
        Internal fit() method that finds the best feature using a feature selector (SelectFromModel or SelectKBest).

        Chooses between SelectKBest (filter-based) or SelectFromModel (model-based)
        based on whether score_func is provided. Fits the selector and records
        which features were selected.

        Parameters
        ----------
        X : array or sparse matrix of shape (n_samples, n_features)
            Validated training features.
        y : array or sparse matrix of shape (n_samples,)
            Validated target values.
        
        Returns
        -------
        self : FeatureSelector
            Fitted instance with _support and _selected_features set.
            
        Raises
        ------
        ValueError
            If score_func is provided without max_features.
            If estimator is None when score_func is None.
            If y is None but required.

        Notes
        -----
        - This method assumes X and y have been validated and converted to NumPy arrays.
        - Sets the following private attributes:
          - _support
          - _selected_features
          - _internal_selector
        - Does not set the public attributes (those ending in _) which are set in fit().
        """

        lg = self._get_logger()

        # Let sklearn validate and set n_features_in_ (private API; silence type checker)
        X_array, y = self._validate_data(  # type: ignore[attr-defined]
            X, y, 
            accept_sparse=True, 
            ensure_2d=True,
            reset=True)
        # check_X_y is done internal to _validate_data - X_array, y = check_X_y(X_array, y, dtype=None) 

        # Handle sparse y (sklearn fit doesn't like it)
        if y is not None and issparse(y):
            y = cast(csr_matrix, y).toarray().ravel()

        # Conditional Selection Logic (SelectFromModel vs. SelectKBest)
        lg.info(f"Creating selector with {self._internal_threshold = }, {self._internal_max_features = }")
        if self._internal_score_func is not None:
            # --- Logic for SelectKBest (Filter-based Selection) ---
            if self._internal_max_features is None:
                lg.critical("score_func requires max_features (k) to be specified.")
                raise ValueError(
                    "When 'score_func' is specified, 'max_features' (k) must also be specified "
                    "to use the filter-based SelectKBest selector."
                )
            if y is None:
                raise ValueError("y cannot be None for supervised feature selection")

            lg.info(
                f"Using SelectKBest with score_func={self._internal_score_func.__name__} and k={self._internal_max_features}"
            )
        
            # Instantiate SelectKBest
            self._internal_selector = SelectKBest(score_func=self._internal_score_func, k=self._internal_max_features)

        else:
            # If user did not provide an estimator but a default internal estimator exists, use it.
            if self.estimator is None and self._internal_estimator is None:
                raise ValueError("estimator must be provided when score_func is None and no internal estimator is available.")

            # Instantiate SelectFromModel using the internal estimator (default) if user did not provide one.
            cloned_estimator = clone(cast(BaseEstimator, self._internal_estimator if self.estimator is None else self.estimator))
            self._internal_selector = SelectFromModel(
                estimator=cloned_estimator,
                threshold=self._internal_threshold,
                max_features=self._internal_max_features,
                prefit=self._internal_prefit
            )
            ## Clone to avoid mutating original
            cloned_estimator = clone(cast(BaseEstimator, self._internal_estimator))
            self._internal_selector = SelectFromModel(
                estimator=cloned_estimator,
                threshold=self._internal_threshold,
                max_features=self._internal_max_features,
                prefit=self._internal_prefit
            )
        
        # sklearn's fit() method doesn't accept sparse matrices for the target variable y.
        # Convert y to dense array if it's sparse
        if issparse(y):
            y_sparse = cast(spmatrix, y)
            y_array = np.asarray(y_sparse.toarray()).ravel()  # type: ignore[attr-defined]
        else:
            y_array = np.asarray(y).ravel()

        # Do the fit
        self._internal_selector.fit(X_array, y_array)

        # Record selected_features_
        self._support = self._internal_selector.get_support()
        if  (hasattr(self, '_feature_names_in') and 
                self._feature_names_in is not None):
            self._selected_features = getattr(self, '_feature_names_in')[self._support]
            # Immediately after computing self._selected_features
            if self._selected_features is not None:
                # Ensure we have a 1-D numpy array of plain Python strings
                self._selected_features = np.asarray(self._selected_features, dtype=object)
                # convert each entry to str to avoid type mismatches (np.str_ vs str vs Index name)
                self._selected_features = np.array([str(x) for x in self._selected_features], dtype=object)

            # n_featueres_in is the number of input features (not the number selected)
            #20251111 self._n_features_in = len(self._selected_features) if self._selected_features is not None else 0
            #20251111 lg.info(f"Selected {self._n_features_in = } features from {len(self._feature_names_in) = }")
            #20251111 lg.debug(f"Selected features: {self._selected_features = }")
            #20251111 self.n_features_in_ = self._n_features_in
        else:
            self._selected_features = None
            #20251111 self.n_features_in_ = 0
            lg.debug(f"Input feature names are not available; cannot record selected feature names of {X = }, {self.__dict__ = }.")

        # Record n_features_in_
        if X_array is not None:
            self._n_features_in = X_array.shape[1]
        else:
            self._n_features_in = len(self._feature_names_in) if self._feature_names_in is not None else 0
        # Number of features selected (derived from mask)
        self._n_selected_features = int(np.sum(self._support)) if self._support is not None else 0
        lg.info(f"Selected {self._n_selected_features} features from {self._n_features_in} input features")
        lg.debug(f"Selected features: {self._selected_features = }")
        # SKLearn API: n_features_in_ is the number of features seen at fit time (input dim)
        self.n_features_in_ = int(self._n_features_in)


        return self

    @debug_pipeline_step("FeatureSelector__Transform")
    def _transform(self, X: Union[np.ndarray, spmatrix]) -> Union[np.ndarray, spmatrix]:
        """
        Apply feature selection mask to X.
        
        Internal method that applies the boolean support mask to select columns.
        Works with both dense and sparse matrices.
        
        Parameters
        ----------
        X : array or sparse matrix of shape (n_samples, n_features_in_)
            Data to transform.
        
        Returns
        -------
        X_selected : array or sparse matrix of shape (n_samples, n_features_selected_)
            Data with only selected features.
        """
        lg = self._get_logger()
        X_array= self._validate_data(   # type: ignore[attr-defined]
            X, 
            accept_sparse=True, 
            ensure_2d=True,
            reset=False)
        
        # Ensure support mask is valid
        if not hasattr(self, "_support") or self._support is None:
            raise ValueError("FeatureSelector._support is None — did you call fit()?")

        mask = np.asarray(self._support, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != X_array.shape[1]:
            raise ValueError(
                f"Support mask shape {mask.shape} does not match X {X_array.shape}."
            )

        # Apply selection
        X_selected = X_array[:, mask]
        
        #20251014debug. They all match exactly. lg.debug(f"DBGTransform02: {X_selected = }\n {X_array = },\n {X = }")

        return X_selected

    @debug_pipeline_step("FeatureSelector_Transform")
    def transform(self, X: Union[pd.DataFrame, np.ndarray, pd.Series, spmatrix]) -> Union[pd.DataFrame, pd.Series, np.ndarray, spmatrix]:
        """
        Reduce X to the selected features using the fitted selector.
        Applies the feature selection determined during fit().

        While fit() sets which features to keep, transform() applies that selection.
        Uses attributes set during fit():
        SK/IMBLearn required:
        - feature_names_in_
        - n_features_in_
        - support_
        Private:
        - _feature_names_in
        - _n_features_in
        - _support
        - _selected_features
        to generate a new X with only the selected features.

        Since pd.DataFrame format has more information than a NumPy array, this method
        tries to preserve the DataFrame structure, including column names and index. However, DFs
        cause failures to SKLearn compliance in some cases.

        Parameters
        ----------
        X : {array, sparse matrix} of shape (n_samples, n_features_in_)
            Data to transform. Must have the same number of features as seen during fit().

        Returns
        -------
        X_transformed : {DataFrame, Series, array, sparse matrix}
            Transformed data with only selected features. Type matches input type:
            - DataFrame input → DataFrame output (with selected column names)
            - Array input → Array output
            - Sparse input → Sparse output (avoids densification)
            - Series input → Series output (if single feature selected)

        Raises
        ------
        NotFittedError
            If transform is called before fit.
        ValueError
            If X has wrong number of features.
            If transformed output doesn't match expected shape.
        """
        lg = self._get_logger()
        if __debug__:
            dict_before = {k: id(v) for k, v in self.__dict__.items()}
            lg.debug(f"DBGTransform04a: {dict_before = }")

        # Check if fitting set everything necessary. sklearn's check_is_fitted performs 
        # other tests as well on the quality of the fit.
        self._check_fit_quality()

        X_array: Union[np.ndarray, spmatrix] = np.array([])  # Placeholder for validated X
        X_selected: Union[np.ndarray, spmatrix] = np.array([])  # Placeholder for selected X

        # For sparse matrices, avoid densification
        if issparse(X):
            # Validate sparse matrix
            X_array = check_array(X, accept_sparse=True, ensure_2d=True)

            # Validate feature count
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Sparse matrix has {X_array.shape[1] = } features, but FeatureSelector "
                    f"expected {self.n_features_in_ = } features"
                )

            # Apply selection
            X_selected = self._transform(X_array)

        else:
            # Input Validation
            X_array, _ = self._validate_and_record_input(
                X, 
                None, 
                require_y = False,
                record_metadata=False,
                feature_names=None
                )

            # Per SKLearn compliance, immediately return empty input
            if isinstance(X, pd.DataFrame) and (X.empty or X.shape[1] == 0):
                return X
            elif isinstance(X_array, np.ndarray) and (X_array.size == 0 or X_array.shape[1] == 0):
                return self._generate_output(X_array, index=getattr(X, 'index', None))
            elif isinstance(X, spmatrix) and (X.shape[0] == 0 or X.shape[1] == 0):
                return self._generate_output(X, index=getattr(X, 'index', None))

            # Apply Selection
            lg.debug(f"DBGTransform05: {self._support = }, {X_array.shape = }")
            X_selected = self._transform(X_array)

     #debugged. Identical.            if np.array_equal(X_selected, X_array):
     #debugged. Identical.                lg.debug("DBGTransform03: No change in data")
     #debugged. Identical.            else:
     #debugged. Identical.                lg.debug(f"DBGTransform03: {X_selected = }\n {X_array = }")

        # Determine the index to use for validation DFs. Prefer the original input index, if provided, then fall back to
        # the recorded fit-time index if its length matches, otherwise use a fresh RangeIndex.
        original_idx = getattr(X, 'index', None) if not issparse(X) else None
        validation_index = pd.RangeIndex(start=0, stop=X_array.shape[0], step=1)
        if original_idx is not None and hasattr(original_idx, "__len__") and not callable(original_idx) and len(original_idx) == X_array.shape[0]:
            validation_index = original_idx
        elif hasattr(self, '_X_index') and self._X_index is not None:
            if hasattr(self._X_index, "__len__") and not callable(self._X_index) and len(self._X_index) == X_array.shape[0]:
                validation_index = self._X_index

        # Create DataFrames for validation
        ## Input DF
        X_df = None
        if hasattr(self, 'feature_names_in_') and not isinstance(X_array, spmatrix) and self.feature_names_in_ is not None and self._selected_features is not None:
            X_df = pd.DataFrame(
                data=X_array,
                columns=self.feature_names_in_,
                index=validation_index
            )
            # Restore dtypes if available
            if hasattr(self, '_X_dtypes') and self._X_dtypes is not None:
                for col in X_df.columns:
                    if col in self._X_dtypes:
                        try:
                            X_df[col] = X_df[col].astype(self._X_dtypes[col])
                        except Exception as e:
                            lg.warning(f"Error applying dtype to input column {col}: {e}")
            self._validator.validate_frame(X_df, "feature_selector_transform_X_input")
        else:
            lg.warning("Skipping input validation DF: feature_names_in_ not available")

        ## Output DF (using selected columns and selected data)
        X_df_selected = None

        ### Refine the columns if available, otherwise keep all the original
        if self._selected_features is not None:
            cols = self._selected_features
        elif X_df is not None:
            cols = X_df.columns
        else:
            cols = [f"feature{i}" for i in range(0, X_selected.shape[1])]
        if not isinstance(X_selected, spmatrix):
            X_df_selected = pd.DataFrame(
                data=X_selected,
                columns=cols,
                index=validation_index
            )
        ### Optionally restore dtypes for selected columns
        if hasattr(self, '_X_dtypes') and self._X_dtypes is not None and X_df_selected is not None:
            for col in X_df_selected.columns:
                if col in self._X_dtypes:
                    try:
                        X_df_selected[col] = X_df_selected[col].astype(self._X_dtypes[col])
                    except Exception as e:
                        lg.warning(f"Error applying dtype to output column {col}: {e}")
        # Validate output DF
            self._validator.validate_frame(X_df_selected, "feature_selector_transform_X_output")
         
        # Compare input and output DFs for consistency (relaxed: only check columns and values, ignore index)
            if X_df is not None:
                # Only check that the selected columns and their values match, ignore index
                input_selected = X_df[self._selected_features] if self._selected_features is not None else X_df
                output_selected = X_df_selected[self._selected_features] if self._selected_features is not None else X_df_selected
                # Compare values and columns, ignore index
                if not (list(input_selected.columns) == list(output_selected.columns) and
                        np.allclose(input_selected.values, output_selected.values, equal_nan=True)):
                    raise ValueError("Transformed output does not match expected selected features (relaxed check):\n "
                                     f"input columns: {input_selected.columns.tolist()}\n "
                                     f"output columns: {output_selected.columns.tolist()}\n "
                                     f"input values: {input_selected.values}\n "
                                     f"output values: {output_selected.values}\n ")
            else:
                lg.warning("Skipping output vs input comparison validation because DataFrame of input X could not be created.")

        # Check for unintended state changes
        dict_after = {k: id(v) for k, v in self.__dict__.items()}
        changed = {k: (dict_before[k], dict_after[k]) for k in dict_before
                   if k in dict_after and dict_before[k] != dict_after[k]}
        if changed:
            lg.error(f"DBGTransform04b: FeatureSelector mutated attributes in transform: {changed}")


        # Validate output
        ## This had to be scrapped because SKLearn's compliance check sends different 
        ## data to fit() and transform(). 
        ## There is no way to make the data provide in fit() match the data in 
        ## transform().
        ## Compare with golden_df if available
        if False:  # hasattr(self, 'golden_df') and self._selected_features is not None:
            expected = self.golden_df[self._selected_features].values

            ### Convert to dense if sparse
            if issparse(X_selected):
                X_sp = cast(spmatrix, X_selected)
                X_selected_dense = np.asarray(X_sp.todense())
            else:
                X_selected_dense = X_selected
            lg.debug(f"DBGTransform01: {X_selected_dense = }\n {expected = }")  
            if not np.array_equal(expected, X_selected_dense):  # type: ignore[attr-defined]
                raise ValueError("Transformed output does not match expected selected features:\n "
                                 f"{expected.shape = }\n "
                                 f"{X_selected_dense.shape = }\n "
                                 f"{expected.dtype = }\n "
                                 f"{type(X_selected_dense) = }\n "
                                 f"{expected = }\n "
                                 f"{X_selected_dense = }\n "
                )

        # Generate output in the original input format
        # Pass the original input index if available (for DataFrame predict/transform)
        original_idx = getattr(X, 'index', None) if (not issparse(X)) else None
        X_res = self._generate_output(X_selected, index=original_idx)

        lg.debug(f"Feature selection: {X_array.shape if hasattr(X_array, 'shape') else 'sparse'} -> {X_res.shape if hasattr(X_res, 'shape') else 'sparse'}")
    #    debugged         if __debug__:
    #    debugged             new_keys = set()
    #    debugged             changed_keys = set()
    #    debugged             dict_after = self.__dict__
    #    debugged             lg.debug(f"DBGTransform04b: {dict_after = }")
    #    debugged             for k, v in dict_after.items():
    #    debugged                 if k not in dict_before:
    #    debugged                     new_keys.add(k)
    #    debugged                 elif dict_before[k] != v:
    #    debugged                     changed_keys.add(k)
    #    debugged             if new_keys:
    #    debugged                 lg.debug(f"DBGTransform04c: transform() added attributes: {new_keys}")
    #    debugged             elif changed_keys:
    #    debugged                 lg.debug(f"DBGTransform04c: transform() changed attributes: {changed_keys}")
    #    debugged             else:
    #    debugged                 lg.debug("DBGTransform04c: No new attributes added during transform(); nothing changed either.")
    #    debugged 

        return X_res
    
    def fit_transform(
            self,
            X: Union[pd.DataFrame, np.ndarray, pd.Series, spmatrix], 
            y: Optional[np.ndarray] = None
            ) -> Union[pd.DataFrame, pd.Series, np.ndarray, spmatrix]:
        """
        Fit to data, then transform it.
        
        Convenience method equivalent to calling fit(X, y).transform(X).
        
        Parameters
        ----------
        X : {DataFrame, array, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        
        Returns
        -------
        X_transformed : {DataFrame, array, sparse matrix}
            Transformed data in same format as input.
        
        See Also
        --------
        fit : Fit the feature selector.
        transform : Transform data using fitted selector.
        """
        fitted_selector = self.fit(X, y)
        X_transformed = fitted_selector.transform(X)

        # Error Checking
        if not hasattr(fitted_selector, 'transform'):
            raise AttributeError("Fitted selector does not have a 'transform' method.")

        return X_transformed
    
    def get_support(self, indices=False) -> np.ndarray:
        """
        Get boolean mask or integer indices of selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return integer indices of selected features.
            If False, return boolean mask.
        
        Returns
        -------
        support : ndarray
            If indices=False: Boolean array of shape (n_features_in_,)
                where True indicates selected feature.
            If indices=True: Integer array of selected feature indices.
        
        Raises
        ------
        NotFittedError
            If called before fit.
        
        Examples
        --------
        >>> selector.fit(X, y)
        >>> mask = selector.get_support()  # [True, False, True, False, True]
        >>> indices = selector.get_support(indices=True)  # [0, 2, 4]
        """
        check_is_fitted(self, "_support")
        if indices:
            return np.where(self.support_)[0]
        else:
            return self.support_

class FeatureSelectingClassifier(FeatureSelectorBase, BaseEstimator, ClassifierMixin):
    """
    Classifier that performs feature selection before training.

    This acts as a meta-estimator, chaining the FeatureSelector transformer with
    a final classifier, ensuring that the same features are consistently used
    across fit, predict, and predict_proba steps.

    Parameters
    ----------
    estimator : estimator object
        The base estimator (classifier) to be used for the final prediction.
        It is also used by the internal FeatureSelector for selection.
        
    max_features : int, optional
        The maximum number of features to select. Passed to FeatureSelector.
        
    threshold : str or float, default='median'
        The threshold value for feature selection. Passed to FeatureSelector.
        
    score_func : callable, optional
        A scoring function for feature selection (e.g., chi2). Passed to FeatureSelector.

    Attributes
    ----------
    feature_selector_ : FeatureSelector
        The fitted feature selector instance.
        
    classifier_ : BaseEstimator
        The fitted final classifier instance.
        
    classes_ : ndarray
        The class labels known to the classifier.
    """

    def __init__(self,
                 estimator: Optional[BaseEstimator] = None,
                 max_features: Optional[int] = None,
                 threshold: Optional[Union[str, float]] = "mean",
                 score_func: Optional[Callable] = None,
                 prefit: bool = False
                ):
        
        self.estimator = estimator
        self.max_features = max_features
        self.threshold = threshold
        self.score_func = score_func
        self.prefit = prefit

    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        if self.estimator is not None and deep:
             # Delegate to the base estimator's get_params
            estimator_params = self.estimator.get_params(deep=deep)
            for k, v in estimator_params.items():
                params[f"estimator__{k}"] = v

        return params

    def set_params(self, **params) -> 'FeatureSelectingClassifier':
        """Set parameters for this estimator."""
        if 'estimator' in params:
            self.estimator = params.pop('estimator')

        # Delegate nested parameters to the estimator
        estimator_params = {k.split('__', 1)[1]: v for k, v in params.items() if k.startswith('estimator__')}
        if estimator_params and self.estimator is not None:
            self.estimator.set_params(**estimator_params)
        
        # Set top-level parameters
        for key, value in params.items():
            if not key.startswith('estimator__') and hasattr(self, key):
                setattr(self, key, value)

        return self

    @debug_pipeline_step("FeatureSelectingClassifier_Fit")
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelectingClassifier':
        """
        Selects features, then fits the final classifier on the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples.
        y : np.ndarray
            The target values.

        Returns
        -------
        self : FeatureSelectingClassifier
            The fitted classifier instance.
        """

        # Initialize internal FeatureSelector and Classifier
        # If no estimator provided, default to LogisticRegression
        _default_estimator = LogisticRegression(max_iter=1000, random_state=42)
        base_est = self.estimator if self.estimator is not None else _default_estimator

        # Private attributes
        self._validator = DataValidator("FeatureSelectingClassifier")
        self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

        self._set_fittedattributes_and_privatevars()

        lg = self._get_logger()

        # 1. Input Validation
        X_array, y_validated =self._validate_and_record_input(X, y, 
                                                    require_y=True, 
                                                    record_metadata=True, 
                                                    feature_names=None)
        self.classes_ = np.unique(np.asarray(y_validated))

        # 2. Initialize and Fit FeatureSelector
        ## Ensure we never ask for more features than exist. If user did not
        ## provide max_features, default to keeping ALL features (robust for tests).
        n_features = X_array.shape[1] if hasattr(X_array, "shape") else None
        if self.max_features is None:
            max_for_selector = n_features
        else:
            # clamp to available features
            max_for_selector = min(self.max_features, n_features) if n_features is not None else self.max_features

        # Use base_est for both the selector and the final classifier
        self.feature_selector_ = FeatureSelector(
            estimator=clone(base_est),  # Selector uses a fresh, un-fitted copy
            max_features=max_for_selector,
            threshold=self.threshold,
            score_func=self.score_func,
        )
        self.classifier_ = clone(base_est)  # Final classifier uses a fresh, un-fitted copy

        lg.debug("Fitting internal FeatureSelector...")
        lg.debug(f"[DBGCV00] Before selector: {X_array.shape = }")

        # Preserve original DataFrame column names for the FeatureSelector
        # by passing the original X when possible so `_feature_names_in` is captured.
        X_selected = self.feature_selector_.fit_transform(X if isinstance(X, pd.DataFrame) else X_array, y_validated)
        
        # Set the fit attributes from the selector to this meta-estimator
        sel = self.feature_selector_
        try:
            self.feature_names_in_ = np.array(sel.feature_names_in_, dtype=object) if hasattr(sel, 'feature_names_in_') and sel.feature_names_in_ is not None else None
            self._feature_names_in = np.array(sel._feature_names_in, dtype=object) if hasattr(sel, '_feature_names_in') and sel._feature_names_in is not None else None
            self.n_features_in_ = getattr(sel, 'n_features_in_', 0)
            self._n_features_in = getattr(sel, '_n_features_in', 0)
            self.support_ = np.array(sel.support_, dtype=bool) if hasattr(sel, 'support_') and sel.support_ is not None else None
            self._support = np.array(sel._support, dtype=bool) if hasattr(sel, '_support') and sel._support is not None else None
            self.selected_features_ = np.array([str(s) for s in sel.selected_features_], dtype=object) if hasattr(sel, 'selected_features_') and sel.selected_features_ is not None else None
            self._selected_features = np.array([str(s) for s in sel._selected_features], dtype=object) if hasattr(sel, '_selected_features') and sel._selected_features is not None else None
        except Exception as e:
            lg.error(f"Error syncing attributes from selector: {e}")
            raise ValueError(f"Attribute sync failed: {e}")

        # 3. Initialize and Fit final Classifier
        lg.debug(f"Fitting final classifier: {type(self.classifier_).__name__}")
        internal_classifier = cast(Any, self.classifier_)

        ## convert DataFrame -> ndarray to avoid dtype/array-function surprises
        X_for_fit = X_selected if isinstance(X_selected, np.ndarray) else np.asarray(X_selected)
        internal_classifier.fit(X_for_fit, np.asarray(y_validated))
        # record the number of input features for sklearn API compliance
        self.n_features_in_ = X_for_fit.shape[1]

        return self

    @debug_pipeline_step("FeatureSelectingClassifier_Predict")
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using selected features."""
        lg = self._get_logger()
        check_is_fitted(self, ['feature_selector_', 'classifier_'])

        # Quick Consistency check - Validate input feature count
        X_checked = check_array(X, ensure_2d=True)
        if X_checked.shape[1] != self.feature_selector_.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but FeatureSelector expected "
                f"{self.feature_selector_.n_features_in_} features."
            )

        # 1. Transform features using the fitted selector
        lg.debug(f"[DBGCV02] During predict: {X_checked.shape = }")
        X_selected = self.feature_selector_.transform(X)
        lg.debug(f"[DBGCV02] During predict: {X_selected.shape = }")

        # 2. Convert to numpy if needed and predict
        X_array = X_selected.values if isinstance(X_selected, pd.DataFrame) else np.asarray(X_selected)
        preds = self.classifier_.predict(X_array)

        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using selected features."""
        check_is_fitted(self, ['feature_selector_', 'classifier_'])

        # Quick Consistency check - Validate input feature count
        X_checked = check_array(X, ensure_2d=True)
        if X_checked.shape[1] != self.feature_selector_.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but FeatureSelector expected "
                f"{self.feature_selector_.n_features_in_} features."
            )

        if not hasattr(self.classifier_, 'predict_proba'):
            raise AttributeError("Underlying classifier does not support predict_proba.")

        # 1. Transform features using the fitted selector
        X_selected = self.feature_selector_.transform(X)

        # 2. Convert to numpy and predict with probabilities
        X_array = X_selected.values if isinstance(X_selected, pd.DataFrame) else np.asarray(X_selected)
        return self.classifier_.predict_proba(X_array)

    