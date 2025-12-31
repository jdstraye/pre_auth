"""
SMOTE sampling component with proper data validation and debugging.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
import sys
import traceback
from typing import List, Optional, Union, Tuple, Dict, Any, Sequence, cast
from imblearn.base import SamplerMixin, BaseSampler
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target
from src.debug_library import debug_pipeline_step, DataValidator
from src.utils import gv, setup_logging

logfile=gv.LOG_DIR / "smote.log"
setup_logging(logfile)
logger = logging.getLogger(__name__)

def convert_DF_to_int(X: pd.DataFrame, y: Union[np.ndarray, pd.Series], headers: Optional[Dict[str, List[str]]] = None) -> tuple[pd.DataFrame, np.ndarray]:
    if headers is not None:
        y_df = pd.DataFrame(y, columns = headers.get('targeted_cols', ["class"]))
    else:
        y_df = pd.DataFrame(y)
    df = pd.concat([X, y_df], axis=1)
    col_indices = list(range(df.shape[1]))
    for idx in col_indices:
        colname = df.columns[idx]
        colval = df[colname]
        # convert to float so np.isfinite works
        bad_mask = ~np.isfinite(colval.astype(float))
        if bad_mask.any():
            logger.critical(f"inf or NaN present in DataFrame['{colname}'] whose values include '{colval.unique()}'")
            raise ValueError("Clean the DataFrame before continuing")
        # safe to cast now
        # use pandas astype with errors='raise' intentionally so we see problems
        df[colname] = colval.astype(int)
    return df.iloc[:,:-1], df.iloc[:,-1].to_numpy()

class BaseNamedSamplerMixin:
    """
    Mixin class providing common functionality for all NamedSMOTE variants,
    including picklable logging and DataFrame input normalization.
    """
    feature_names_in_: np.ndarray

    def __init__(self, *args, **kwargs):
        # Initialize attributes needed by _get_logger which will not write to a file (to be picklable)
        self._logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        
        # ensure a logger can exist and be configured, but DO NOT store the Logger object on self
        _tmp_lg = logging.getLogger(self._logger_name)
        _tmp_lg.setLevel(logging.DEBUG)
        _tmp_lg.propagate = False
        del _tmp_lg

        lg = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        """
        Return a logger by name. Resolve it at call time so we do NOT store the
        Logger object on the instance (keeps the instance pickleable).
        """
        name = getattr(self, "_logger_name", f"{__name__}.PicklableSMOTESampler")
        lg = logging.getLogger(name)
        # keep runtime guarantees: debugging enabled & propagation on
        if lg.level == logging.NOTSET:
            lg.setLevel(logging.DEBUG)
        lg.propagate = False

        # Attach handlers if missing
        if not lg.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter(
                "%(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
            ))
            lg.addHandler(ch)

        return lg  # Always returns a Logger

    def _ensure_dataframe(self, X, feature_names: Optional[List[str]]=None):
        """Convert input X to a DataFrame, preserving column names if available."""
        cols =[]
        X_result = pd.DataFrame()
        index = None
        X_dense = None
        X_array = None

        # Find X
        if isinstance(X, pd.DataFrame):
            return X
        
        if sparse.issparse(X):
            X_dense = X.toarray()
        elif isinstance(X, (np.ndarray, list)): # X is neither a pd.DataFrame nor sparse
            X_array = np.asarray(X)
        else:
            raise ValueError(f"Cannot convert object of type {type(X)} to DataFrame")

        # Find Indices
        if isinstance(X, (pd.DataFrame, pd.Series)):
            index = getattr(X, "index", None)
        if index is None: #including lists, even though they have an index() attribute
            n_rows = 0
            if X_dense is not None:
                n_rows = X_dense.shape[0]
            else:
                if X_array is not None:
                    n_rows = X_array.shape[0]  
                else:
                    if hasattr(X, "shape"):
                        n_rows = getattr(X, "shape")[0] 
                    else:
                        n_rows = 0

            try:
                index = range(n_rows)
            except TypeError:
                index = None  # fallback

        # Find Columns
        ## Priority: feature_names_in_ -> feature_names arg -> self.feature_names -> 
        ##           generic names
        cols = getattr(self, "feature_names_in_", None)
        lg = self._get_logger()
        msg="DBGcol00- " + (
            f"{self.feature_names_in_ = }{cols = }" if hasattr(self, "feature_names_in_") else \
            f"no 'feature_names_in_' attr")
        lg.debug(f"{msg}")
        if cols is None or len(cols) == 0:
            if feature_names is not None:
                cols = list(feature_names)
                lg.debug(f"DBGcol02- {feature_names = }{cols = }")
            elif hasattr(self, "feature_names") and getattr(self, "feature_names", None):
                cols = list(getattr(self, "feature_names"))
                if hasattr(self, "feature_names"):
                    msg=f'DBGcol04- {getattr(self, "feature_names") = }{cols = }'
                else:
                    msg=f"no 'feature_names' attr"
                    lg.debug(f"{msg}")
            else:
                # Generic fallback
                if X_dense is not None: 
                    n_features = X_dense.shape[1] 
                else:
                    if X_array is not None:
                        n_features = X_array.shape[1]
                    else:
                        if hasattr(X, "shape"):
                            n_features = getattr(X, "shape")[1]
                        else:
                            n_features = len(X)
                cols = [f"feature{i}" for i in range(n_features)]
                if hasattr(X, "shape"):
                    msg=f"DBGcol06- {X_dense = }\n{X_array = }\n{getattr(X, 'shape') = }\n{cols = }"
                else:
                    msg=f"{type(X) = }"
                lg.debug(f"{msg}")
        # Save the names for future calls (e.g., in a pipeline)
        setattr(self, "feature_names_in_", cols)

        # Put them together
        X_d = X_dense if X_dense is not None else (X_array if X_array is not None else X)

        X_result = pd.DataFrame(X_d, columns=cols, index=index)
        self._get_logger().debug(f"MaybeSMOTESampler: converted input to DataFrame with columns: {cols}")

        return X_result

class MaybeSMOTESampler(BaseNamedSamplerMixin, BaseSampler):
    """
    A safe wrapper that acts as a sampler step in ImbPipeline. It supports:
      - smote__enabled: bool (if False, passthrough)
      - smote__categorical_feature_names: list of names to use for SMOTENC
      - smote__k_neighbors: int
      - smote__sampling_strategy: passed down to SMOTE(SMOTENC)
      - smote__random_state
      - smote__min_improvement: float min required improvement in minority share after resample (sanity check)
    Behavior:
      - If enabled==False -> returns X, y unchanged.
      - If enabled==True:
          -> If categorical_feature_names contain columns present in X -> NamedSMOTENC
          -> else -> NamedSMOTE
      - After resampling, performs a sanity check: minority share must increase by at least min_improvement.
        If not, raises ValueError to indicate SMOTE didn't have the expected effect.
    Notes:
    - This object is intentionally lightweight and used within ImbPipeline; it exposes set_params/get_params
    so sklearn's set_params pipeline calls work.
     - To be picklable, the following changes were made:
        - Using a local logger, _logger, instead of the global logger, that does not save to a file.
        - Added methods for defining what is/is not pickled: 
           - __getstate__ excludes the _logger from pickling
           - __setstate__ redefines the _logger
        - Added attributes for imblearn pipeline recognition
    """

    # -------------------------------------------------------------
    # class attribute '_parameter_constraints'
    # is required by scikit-learn >= 1.4 for parameter validation.
    # This is a basic set of constraints for the parameters.
    _parameter_constraints: Dict[str, Any] = {
        "enabled": [bool],
        "categorical_feature_names": [list, None],
        "k_neighbors": [int],
        "sampling_strategy": [str, float, dict],
        "random_state": [int, None],
        "min_improvement": [float]
    }

    # More requirements for sklearn compatibility
    _sampling_type = "over-sampling"

    def __init__(self,
                 enabled: Optional[bool] = True,
                 headers: Optional[Dict[str, List[str]]] = {},
                 #feature_names: Optional[List[str]] = None,
                 #categorical_feature_names: Optional[List[str]] = None,
                 #ohe_column_names: Optional[List[str]] = None,
                 k_neighbors: Optional[int] = 5,
                 #20251004 sampling_strategy: Optional[Union[str, float, Dict[str, int]]] = 'auto',
                 sampling_strategy = 'auto',
                 random_state: Optional[int] = gv.RANDOM_STATE,
                 min_improvement: Optional[float] = gv.DEFAULT_SMOTE_MIN_IMPROVEMENT,
                 allow_fallback: bool = True,
                 *args,
                 **kwargs
                ):
        
        # Initialize the mixin first to set _logger_name and define shared methods
        BaseNamedSamplerMixin.__init__(self, *args, **kwargs)

        # Initialize the parent imblearn class
        BaseSampler.__init__(self, sampling_strategy=sampling_strategy)

        # Now, set your own parameters.
        self.enabled = enabled
        if headers:
            self.feature_names = headers.get('feature_cols', [])
            self.categorical_feature_names = headers.get('categorical_cols', [])
            self.ohe_column_names = headers.get('ohe_cols', [])
            self.taget_column_names = headers.get('target_cols', [])
        else:
            self.feature_names = []
            self.categorical_feature_names = []
            self.ohe_column_names = []
            self.taget_column_names = []
        self.k_neighbors = int(k_neighbors or 0)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.min_improvement = float(min_improvement or 0.0)
        # Whether the sampler should gracefully fallback to a safe oversampler
        # (RandomOverSampler) when SMOTE cannot be applied due to too-few
        # minority examples or incompatible encoding.
        self.allow_fallback = bool(allow_fallback)

        # Tracking attributes for diagnostics (set per-call)
        self.last_min_class_count: Optional[int] = None
        self.last_smote_applied: bool = False
        self.last_fallback_used: bool = False
        self.last_used_sampler: Optional[str] = None
        #20251004self._validator = DataValidator("MaybeSMOTESampler", X=X, y=y, categorical_feature_names=self.categorical_feature_names, ohe_column_names=self.ohe_column_names)

    # scikit-learn compatibility
    def get_params(self, deep=True):
        """
        Returns parameters for sklearn pipeline compatibility.
        """
        return {
            "enabled": self.enabled,
            "categorical_feature_names": self.categorical_feature_names or None,
            "k_neighbors": self.k_neighbors,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
            "min_improvement": self.min_improvement
        }

    def set_params(self, **params):
        """
        Sets parameters for sklearn pipeline compatibility.
        Supports 'smote__...' style keys from ImbPipeline.
        """        
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Accept 'smote__...' style if needed (pipeline will pass direct names)
                setattr(self, k, v)
        return self
    
    def _class_counts(self, y: Any) -> Dict[Any, int]:
        """
        Counts the occurrences of each unique class in the input sequence.
        
        Parameters
        ----------
        y : Union[Sequence, pd.Series]
            An array-like object containing the class labels.
        
        Returns
        -------
        Dict[Any, int]
            A dictionary where keys are the class labels and values are their counts.
        """
        # Ensure y is a pandas Series for a robust value_counts operation.
        # Flattens y if it's a multi-dimensional array-like object.
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.ravel()
            
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y)
        else:
            y_series = y
        return y_series.value_counts().sort_index().to_dict()

    def _minority_share(self, counts: Dict[Any, int]) -> float:
        """
        Computes the fraction of samples in the minority class (lowest count / total).
        
        Args:
            counts: Dict of class labels to counts.
        
        Returns:
            Float of minority share; 0.0 if counts empty.
        """
        # fraction of samples that belong to the minority class (lowest count / total)
        if not counts:
            return 0.0
        total = sum(counts.values())
        mn = min(counts.values())
        return mn / float(total) if total > 0 else 0.0

    def _get_categorical_indices(self, X: pd.DataFrame) -> List[int]:
        """Get column indices for categorical features."""
        indices = []
        for cat_name in self.categorical_feature_names:
            if cat_name in X.columns:
                idx = X.columns.get_loc(cat_name)
                indices.append(idx)
        return indices

    def _adjust_k_neighbors(self, y: Union[np.ndarray, pd.Series]) -> int:
        """Adjust k_neighbors to be valid for the minority class size."""
        if isinstance(y, pd.Series):
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)
        y_arr = y_arr.astype(int)

        class_counts = np.bincount(y_arr)
        min_class_size = class_counts.min()

        max_k = max(1, min_class_size - 1)
        if self.k_neighbors >= min_class_size:
            logger.warning(f"k_neighbors={self.k_neighbors} too large for min class size={min_class_size}, using k={max_k}")
            return max_k
        return self.k_neighbors

    def _validate_improvement(self, y_original: np.ndarray, y_resampled: np.ndarray) -> None:
        """Validate that SMOTE improved class balance sufficiently."""
        def minority_share(y_arr):
            counts = np.bincount(y_arr)
            return counts.min() / counts.sum()

        original_share = minority_share(y_original)
        new_share = minority_share(y_resampled)
        improvement = new_share - original_share

        logger.info(f"SMOTE improvement: {original_share:.3f} -> {new_share:.3f} (+{improvement:.3f})")

        if improvement < self.min_improvement:
            raise ValueError(f"SMOTE improvement {improvement:.3f} below minimum {self.min_improvement}")

    def _validate_y(self, y):
        """
        Validate and normalize target labels for sklearn/imblearn compliance.

        This method ensures that the input target array `y` conforms to one of the
        supported formats required by sklearn and imblearn sampling estimators. It
        also stores the detected format in `self._y_format` to guide output formatting
        in `_fit_resample`.

        Supported formats:
            1. 1D array of shape (n_samples,) containing class labels
            2. 2D column vector of shape (n_samples, 1) containing class labels
            3. 2D one-hot encoded array of shape (n_samples, n_classes) where:
               - All entries are binary (0 or 1)
               - Each row sums to exactly 1

        Unsupported formats (will raise ValueError):
            - Probability distributions (values not in {0, 1})
            - Continuous targets
            - Multi-output targets (y.shape[1] > 1 and not one-hot)
            - Any other non-conforming format
        
        Here are some cases for y formats and what should be done -
        | Case | Y format | sampler |
        | ---- | -------- | ------- |
        |  1   | 1D y (shape (n_samples,)) | must work. |
        |  2   | 2D (n_samples, 1) column vector of labels | must work. |
        |  3   | 2D one-hot (shape (n_samples, n_classes), entries in {0,1}, row sum = 1) | allowed, collapse via argmax |
        | other| Anything else (probabilities, continuous, mixed, multioutput) | must raise ValueError.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values to validate.

        Returns
        -------
        y_validated : ndarray
            Validated target array. For case 1 and 2, returns 1D array of shape
            (n_samples,). For case 3, returns the original 2D one-hot array.

        Raises
        ------
        ValueError
            If `y` is not in one of the supported formats.

        Notes
        -----
        Sets `self._y_format` to one of: 'multiclass', 'onehot', or 'unsupported'.
        This attribute is used by `_fit_resample` to determine the output format.
        """
        y = np.asarray(y)

        # Case 1: 1D labels are fine
        if y.ndim == 1:
            self._y_format = "multiclass"
            return y

        # Case 2: (n_samples, 1) column vector of labels
        if y.ndim == 2 and y.shape[1] == 1:
            self._y_format = "multiclass"
            return y.ravel()

        # Case 3: one-hot encoded
        if y.ndim == 2 and np.all((y == 0) | (y == 1)) and np.all(y.sum(axis=1) == 1):
            self._y_format = "onehot"
            return y.argmax(axis=1)   # 1D labels

        # Otherwise: reject
        self._y_format = "unsupported"
        raise ValueError(
            f"Unsupported label format y with shape {y.shape} and type_of_target={type_of_target(y)}. "
            "MaybeSMOTESampler only supports 1D class labels or one-hot encoded labels."
        )
    
    def fit_resample(self,  X: Union[pd.DataFrame, np.ndarray, csr_matrix, list],
                            y: Union[pd.DataFrame, pd.Series, np.ndarray, list]
                    ) -> Tuple[ Union[pd.DataFrame, np.ndarray, csr_matrix, list],
                                Union[pd.DataFrame, pd.Series, np.ndarray, list]]:
        """
        Public API entry point. Ensures sklearn/imblearn compliance.
        Resample the dataset with input/output format preservation.

        This method serves as a format-aware wrapper around `_fit_resample` to ensure
        sklearn and imblearn API compliance. It preserves the input format of both X
        and y through the resampling process.

        Format preservation rules:
            - Sparse X input → sparse X output (csr_matrix)
            - List X input → list X output  
            - DataFrame X input → DataFrame X output
            - ndarray X input → ndarray X output
            - Series y input → Series y output (with preserved name and new index)
            - DataFrame y input → DataFrame y output (with preserved columns and new index)
            - List y input → list y output
            - ndarray y input → ndarray y output

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame, list} of shape (n_samples, n_features)
            Training data to be resampled.
        y : {array-like, Series, DataFrame, list} of shape (n_samples,) or (n_samples, n_outputs)
            Target values. Supports:
            - 1D class labels
            - 2D column vector of shape (n_samples, 1)
            - 2D one-hot encoded labels (binary values, row sum = 1)

        Returns
        -------
        X_resampled : {ndarray, DataFrame, csr_matrix, list}
            Resampled feature data in the same format as input X.
        y_resampled : {ndarray, Series, DataFrame, list}
            Resampled target values in the same format as input y.

        Raises
        ------
        ValueError
            If y is not in a supported format (see `_validate_y` for details).

        Notes
        -----
        Internally calls `_validate_y` to normalize y to a 1D label array, then
        delegates to `_fit_resample` which returns (DataFrame, ndarray). This method
        then converts the output back to match the input format.
        """
        y_valid = self._validate_y(y)
        X_res, y_res = self._fit_resample(X, y_valid)

        # Track X
        X_is_list = isinstance(X, list) 
        self._input_type = "dataframe" if isinstance(X, pd.DataFrame) else \
                           "sparse" if sparse.issparse(X) else \
                           "list" if X_is_list else \
                           "ndarray"

        # Track y
        y_is_series = isinstance(y, pd.Series)
        y_is_dataframe = isinstance(y, pd.DataFrame)
        y_is_list = isinstance(y, list) 

        # Restore X
        if self._input_type == "sparse":
            X_final = sparse.csr_matrix(X_res.values)
        elif self._input_type == "list":
            # If input was list, return list of lists
            X_final = X_res.values.tolist()
        elif self._input_type == "ndarray":
            X_final = X_res.values
        else:  # dataframe
            X_final = X_res

        # Restore Y
        y_final = y_res # Base is 1D np.ndarray

        if y_is_dataframe:
            # If the y argument was a DataFrame, the output y must be a DataFrame
            # Restore to DataFrame with a generic column name 'class' (or use y.columns if possible)
            if hasattr(y, 'columns') and len(y.columns) == 1:
                 # Use the original column name if it was a single-column DataFrame
                 y_final = pd.DataFrame(y_res, columns=y.columns, index=X_res.index)
            else:
                 # Fallback to a generic column name
                 y_final = pd.DataFrame(y_res, columns=['target'], index=X_res.index)

        elif y_is_series:
            # If the input was a Series, the output should be a Series
            y_final = pd.Series(y_res, name=y.name, index=X_res.index)

        elif y_is_list:
            # If y was a list, return a simple flat list
            y_final = y_res.tolist()

        # If y was a numpy array, y_res (np.ndarray) is already correct.

        return X_final, y_final

    @debug_pipeline_step("MaybeSMOTESampler-_fit_resample")
    def _fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Internal method to perform SMOTE resampling with standardized output format.

        This method handles the core resampling logic by delegating to either
        NamedSMOTE or NamedSMOTENC based on whether categorical features are present.
        It always returns data in a standardized format (DataFrame, ndarray) regardless
        of input format.

        The method performs several key operations:
            1. Converts X to DataFrame if needed
            2. Normalizes y to 1D label array
            3. Validates data integrity before resampling
            4. Adjusts k_neighbors if necessary based on minority class size
            5. Delegates to appropriate SMOTE implementation
            6. Validates and converts results back to integers where appropriate
            7. Performs sanity checks on minority class improvement

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature data. Will be converted to DataFrame internally.
        y : array-like of shape (n_samples,)
            The target labels as a 1D array. Should be the validated output from
            `_validate_y`, ensuring proper format and setting `self._y_format`.

        Returns
        -------
        X_resampled : pd.DataFrame of shape (n_samples_new, n_features)
            The resampled feature data as a DataFrame with preserved column names.
        y_resampled : np.ndarray of shape (n_samples_new,)
            The resampled target labels as a 1D numpy array.

        Raises
        ------
        ValueError
            - If X and y have mismatched lengths
            - If minority share improvement is less than `min_improvement`
            - If multilabel/multioutput targets are detected
            - If data validation fails

        Notes
        -----
        - If `enabled=False`, performs pass-through without resampling
        - Categorical columns are converted back to integers after SMOTE interpolation
        - Copies fitted attributes (sampling_strategy_, n_features_in_) from delegate
        - Extensive logging of before/after statistics and data quality checks
        """
        # If disabled, pass-through with safe types
        if not self.enabled:
            self._get_logger().debug("MaybeSMOTESampler: disabled - passthrough")
            X_df = self._ensure_dataframe(X)
            return X_df, np.asarray(y).ravel()

        lg = self._get_logger()
        lg.debug(f"[LOGGER STATE] name={lg.name} effective={logging.getLevelName(lg.getEffectiveLevel())} handlers={len(lg.handlers)} propagate={lg.propagate}")
        lg_root = logging.getLogger()
        lg_root.debug(f"[ROOT HANDLERS] count={len(lg_root.handlers)}  root_level={logging.getLevelName(lg_root.level)}")

        # Always work with a DataFrame
        X = self._ensure_dataframe(X)
        X_df = X.copy()

        #20251004 # Flatten y to 1D before creating the pandas Series
        #20251004 y_flat = np.asarray(y).ravel()
        #20251004 y_series = pd.Series(y_flat)
        
        # Normalize y into a 1-D label vector
        arr_y = np.asarray(y)

        # Prefer a robust conversion for common formats:
        # - 1-D labels: keep as-is
        # - 2-D one-hot / indicator or probability vectors: convert via argmax(axis=1)
        # - 2-D column vector (n_samples, 1): squeeze
        # - Otherwise fall back to ravel() (but warn)
        if arr_y.ndim == 1:
            y_flat = arr_y.ravel()
        elif arr_y.ndim == 2:
            # If shapes look transposed (rare), attempt to transpose to match X rows
            if arr_y.shape[0] != X.shape[0] and arr_y.shape[1] == X.shape[0]:
                self._get_logger().warning(
                    f"y has shape {arr_y.shape} which looks transposed relative to X {X.shape}; transposing y"
                )
                arr_y = arr_y.T

            if arr_y.shape[0] == X.shape[0] and arr_y.shape[1] > 1:
                # likely one-hot/indicator or probability vector; prefer argmax
                # check if it's a strict one-hot (0/1 rows summing to 1)
                row_sums = arr_y.sum(axis=1)
                if np.all((arr_y == 0) | (arr_y == 1)) and np.all(row_sums == 1):
                    # Case A: one-hot multiclass
                    y_flat = arr_y.argmax(axis=1)

                elif np.allclose(row_sums, 1) and not np.all((arr_y == 0) | (arr_y == 1)):
                    # Case B: probability vectors
                    self._get_logger().warning(
                        "y appears to be probability vectors; converting to 1-D via argmax(axis=1)."
                    )
                    y_flat = arr_y.argmax(axis=1)

                else:
                    # Case C: multilabel/multioutput → not supported
                    raise ValueError("Multilabel and multioutput targets are not supported.")
            elif arr_y.shape[0] == X.shape[0] and arr_y.shape[1] == 1:
                y_flat = arr_y.ravel()
            else:
                # last resort: ravel (this produced the earlier bug)
                self._get_logger().warning(
                    f"Unexpected y shape {arr_y.shape}; flattening with ravel(). If this was a one-hot matrix, use shape (n_samples, n_classes)."
                )
                y_flat = arr_y.ravel()
        else:
            # higher-dim arrays -> flatten but warn
            self._get_logger().warning(f"y has unexpected ndim={arr_y.ndim}; flattening with ravel()")
            y_flat = arr_y.ravel()

        y_series = pd.Series(y_flat)

        self._validator = DataValidator("MaybeSMOTESampler", X=X_df, y=y_flat, categorical_feature_names=self.categorical_feature_names, ohe_column_names=self.ohe_column_names)

        self._get_logger().debug(f"Frame Validation before SMOTE ...")
        self._validator.validate_frame(X_df, "SMOTE")

        orig_counts = self._class_counts(y_series)
        orig_min_share = self._minority_share(orig_counts)

        # Adjust k_neighbors if too large
        self.k_neighbors = self._adjust_k_neighbors(y_flat)

        # Confirm X >< y correlation
        if X_df.shape[0] != len(y_series):
            raise ValueError(f"X and y length mismatch: {X_df.shape[0] = } vs {len(y_series) = }")

        # Build concrete sampler (explicit delegation)
        # Decide whether to use SMOTE/SMOTENC or fallback to RandomOverSampler.
        # If minority class size is too small relative to k_neighbors, prefer fallback
        orig_counts_pre = self._class_counts(y_series)
        min_class_size_pre = min(orig_counts_pre.values()) if orig_counts_pre else 0
        if self.allow_fallback and (min_class_size_pre <= 1 or min_class_size_pre <= int(self.k_neighbors)):
            self._get_logger().warning(f"Using RandomOverSampler fallback: min_class_size={min_class_size_pre}, k_neighbors={self.k_neighbors}")
            sampler = RandomOverSampler(random_state=self.random_state)
            self.last_fallback_used = True
            self.last_used_sampler = 'RandomOverSampler'
            self.last_smote_applied = False
        else:
            if self.categorical_feature_names:
                sampler = NamedSMOTENC(
                    feature_names=self.feature_names,
                    categorical_feature_names=self.categorical_feature_names,
                    k_neighbors=int(self.k_neighbors),
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                )
                self.last_used_sampler = 'NamedSMOTENC'
                self.last_smote_applied = True
            else:
                sampler = NamedSMOTE(
                    feature_names=self.feature_names,
                    k_neighbors=int(self.k_neighbors),
                    sampling_strategy=str(self.sampling_strategy),
                    random_state=self.random_state,
                )
                self.last_used_sampler = 'NamedSMOTE'
                self.last_smote_applied = True
#debug20250917            sampler = NamedSMOTE(
#debug20250917                categorical_feature_names=[],
#debug20250917                k_neighbors=int(self.k_neighbors),
#debug20250917                sampling_strategy=self.sampling_strategy,
#debug20250917                random_state=self.random_state,
#debug20250917            )

        # Save orig counts for sanity check
        orig_counts = self._class_counts(y_series)
        orig_min_share = self._minority_share(orig_counts)
        orig_total_samples = len(y_series)
        orig_minority_samples = min(orig_counts.values())

        feature_names = self.feature_names if self.feature_names else list(X_df.columns)

        self._get_logger().debug(f"{X_df.dtypes.to_dict() = }")

        X_res = None
        y_res = None

        # Delegate to the concrete sampler (this will not call back into this class)
        try:
            # Copy the fitted attributes expected by imblearn checks
            self.sampling_strategy_ = getattr(sampler, "sampling_strategy_", None)
            self.n_features_in_ = getattr(sampler, "n_features_in_", X_df.shape[1])
            fni_exists = (hasattr(self, "feature_names_in") and len(self.feature_names_in_) > 0)
            if not fni_exists:
                if (self.feature_names is not None and len(self.feature_names) > 0):
                    self.feature_names_in_ = np.array(self.feature_names, dtype=object)
                else:
                    #self.feature_names_in_ = np.array(X_df.columns, dtype=object)
                    self.feature_names_in_ = np.array(X_df.columns, dtype=object)


            self._get_logger().debug(f"Applying sampler: {sampler.__class__.__name__}")
            result = sampler.fit_resample(X_df, y_series) # Pass DataFrame and Series

            self._get_logger().debug(f"Sampler applied: {sampler.__class__.__name__}")

            if len(result) == 2:
                X_res, y_res = result
                # restore one-hot if that’s what came in
                if self._y_format == "onehot":
                    classes = np.unique(y)  # preserve original classes
                    y_res = label_binarize(y_res, classes=classes)
                return self._ensure_dataframe(X_res), np.asarray(y_res)
            
        except Exception as e:
            self._get_logger().error(f"The sampler crashed.\nError: '{e}'")
            self._get_logger().error(traceback.format_exc())
            raise

        # Convert back to DataFrame if sampler returned ndarray and we want column names
        if not isinstance(X_res, pd.DataFrame):
            X_res = self._ensure_dataframe(X_res)
        y_res = np.asarray(y_res).ravel()

        # Check results
        self._validator.validate_frame(X_res, "MaybeSMOTESampler")
        self._validator.compare_frames(X_df, X_res, "MaybeSMOTESampler")

        # Convert categorical columns back to integers after SMOTE(SMOTENC)
        # SMOTE(SMOTENC) does a multidimensional interpolation across all 
        # features/classes that may convert integers columns to float during 
        # processing. The BKM is just to cast them back to int.
        convert_DF_to_int(X = X_res, y = y_res)

        # Calculate final counts for summary
        new_counts = self._class_counts(pd.Series(y_res))
        new_min_share = self._minority_share(new_counts)
        new_total_samples = len(y_res)
        new_minority_samples = min(new_counts.values())

        # Sanity check
        new_counts = self._class_counts(pd.Series(y_res))
        new_min_share = self._minority_share(new_counts)
        self._get_logger().info(f"SMOTE applied: orig_counts={orig_counts}, new_counts={new_counts}")
        if self.min_improvement and (new_min_share/orig_min_share)-1 < float(self.min_improvement):
            raise ValueError(
                f"SMOTE sanity check failed: minority share improvement {new_min_share = }, {orig_min_share = }, {(new_min_share/orig_min_share) - 1 = } "
                f"is less than required {self.min_improvement:.6f}. {orig_counts = }, {new_counts = }"
            )
        
        # Create and log a comprehensive summary
        nans_in_output = None
        dtype_changes = None
        summary = (
            f"SMOTE applied successfully.\n"
            f"  Original Sample Counts: {orig_counts} (Total: {orig_total_samples})\n"
            f"  New Sample Counts: {new_counts} (Total: {new_total_samples})\n"
            f"  New Samples Added: {new_total_samples - orig_total_samples} "
            f"({new_minority_samples - orig_minority_samples} minority samples)\n"
            f"  Minority Share Improvement: {(new_min_share - orig_min_share):.4f} "
            f"(Min required: {self.min_improvement:.4f})\n"
            f"  Columns with NaNs in output: {nans_in_output if nans_in_output else 'None'}\n"
            f"  Columns with dtype changes: {dtype_changes if dtype_changes else 'None'}"
        )

        self._get_logger().info(summary)

        # Do a final integrity check
        final_smote_integrity = DataValidator(
            name = "MaybeSMOTESampler-final integrity",
            X = X_res,
            y = y_res,
            categorical_feature_names = self.categorical_feature_names,
            ohe_column_names = self.ohe_column_names
        )
        self._get_logger().debug(f"Frame Validation after SMOTE")
        integrity = final_smote_integrity.validate_frame(df=X_res, stage="SMOTE")
        self._get_logger().debug(f"{integrity = }")
        


        # Return DataFrame and ndarray
        return X_res, np.asarray(y_res)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation. SKLearn required method.

        This method provides feature names for the output of the sampler, which are
        identical to the input feature names since SMOTE resampling does not modify
        or create new features—it only adds synthetic samples.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, generic names of the form "feature0",
            "feature1", ..., "featureN" are generated, where N = n_features_in_ - 1.
            If provided, must be an array-like of length equal to `n_features_in_`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Output feature names. If `input_features` is None, returns generated
            feature names. Otherwise, returns the provided `input_features` unchanged.

        Raises
        ------
        NotFittedError
            If the sampler has not been fitted yet (no `n_features_in_` attribute).
        ValueError
            If `input_features` is provided but its length does not match
            `n_features_in_`.
            If `input_features` is provided but its content does not match
            `feature_names_in_`.

        Examples
        --------
        >>> sampler = MaybeSMOTESampler()
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 0, 1])
        >>> sampler.fit_resample(X, y)
        >>> sampler.get_feature_names_out()
        array(['feature0', 'feature1'], dtype=object)

        >>> sampler.get_feature_names_out(['age', 'income'])
        array(['age', 'income'], dtype=object)

        Notes
        -----
        This method follows the sklearn transformer API convention. Since SMOTE
        resampling adds synthetic samples without modifying features, the output
        feature names are always identical to the input feature names.
        """
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError("This MaybeSMOTESampler instance is not fitted yet.")

        n_features = self.n_features_in_

        if input_features is None:
            return self.feature_names_in_
        
        input_features = np.asarray(input_features, dtype=object)

        if input_features.shape[0] != n_features:
            raise ValueError(
                f"input_features should have length equal to n_features_in_:"
                f"\n {self.n_features_in_ = }, got {len(input_features) = }"
            )
                    
        lg = self._get_logger()
        lg.debug(f"DBGfno0:\n - {self.feature_names_in_ = }\n - {input_features = } ")
        if not np.array_equal(input_features, np.asarray(self.feature_names_in_)):
            raise ValueError(
                f"input_features is not equal to feature_names_in_:"
                f"\n {input_features = }"
                f"\n {self.feature_names_in_ = }"
            )

        return input_features

    def _more_tags(self):
            """
            Tags for imblearn pipeline compatibility, marking this as a sampler.
            """
            return {'stateless': True, 'requires_y': True, 'sampler': True}

class NamedSMOTE(BaseNamedSamplerMixin, SMOTE):
    """
    A SMOTE over-sampler that handles features by name.

    This class extends the standard `imbalanced-learn` SMOTE to accept
    feature names directly but only handles numerical features. 
    It is a robust solution for numerical data within a pandas DataFrame. It maps
    the provided names to their corresponding column indices during
    the `fit_resample` process.
    """
    def __init__(self, 
                 feature_names: Optional[List[str]] = None, 
                 **kwargs):
        """
        Initializes the NamedSMOTE over-sampler.
        This class ignores the `categorical_feature_names` parameter except to create
         a data frame-friendly interface, as it is designed for purely numerical data.
        
        Parameters
        ----------
        categorical_feature_names : list of str
            A list of column names corresponding to the data.
            This argument is optional.
            
        **kwargs : dict
            Additional keyword arguments to be passed to the parent `SMOTE` class.

        Raises
        ------
        TypeError
            If the input `X` is not a pandas DataFrame.
        """
        # Run base initializations
        BaseNamedSamplerMixin.__init__(self, **kwargs)
        SMOTE.__init__(self, **kwargs)

        # initialize with placeholder; we'll set indices at fit_resample
        self.feature_names = feature_names or []

    def _fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Resamples the dataset, mapping categorical feature names to indices.

        This method ensures the input `X` is a pandas DataFrame, then maps the
        provided `categorical_feature_names` to their integer column indices.
        These indices are then passed to the parent `SMOTE` class for
        resampling. The method also handles the reconstruction of the output
        into a DataFrame with the original column names, ensuring metadata
        is preserved.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data, which must be a DataFrame.
        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        X_resampled : pandas.DataFrame
            The resampled feature matrix.
        y_resampled : array-like of shape (n_samples_new,)
            The corresponding resampled target labels.

        Raises
        ------
        ValueError
            If a column name maps to multiple columns or an unexpected location type.
        """
        self._get_logger().debug(f"DBG:SMOTERECUR00 - ENTER {self.__class__.__name__}.fit_resample")

        # If input is not a DataFrame, try to convert it to DataFrame
        if not isinstance(X, pd.DataFrame):
            X = self._ensure_dataframe(X)

        X_df = X.copy()
        if not hasattr(X_df, "columns"):
            raise TypeError("NamedSMOTE expects a pandas DataFrame.")

        self._get_logger().debug(f"DBG:SMOTERECUR01 FEATURES -\n {self.feature_names = }\n {X.columns = }")

        # Convert to arrays for parent SMOTENC but keep X for column names
        X_arr = np.asarray(X) 
        y_arr = np.asarray(y)
        logging.debug(f"{type(X_arr) = }, {type(y_arr) = }")
        logging.debug(f"{type(X) = }, {type(y) = }")

        logger.debug(f"DBG:SMOTERECUR01 - pre-super {self.__class__.__name__}.fit_resample")
        logger.debug(f"DBG:SMOTERECUR01a - {self.__class__.__name__}\n{X_arr = }\n{y_arr = }")

        logger.debug(f"DBG:SMOTERECUR02 - post-super {self.__class__.__name__}.fit_resample")

        X_res, y_res = None, None
        try:
            #debug20250917 result = real_fit_resample(self, X_arr, y_arr)  # avoid recursion
            result = super()._fit_resample(X_arr, y_arr)
            if len(result) == 2:
                X_res, y_res = result
            else:
                raise ValueError("Unexpected return value from fit_resample.")
        except Exception as e:
            # Log full context and traceback
            self._get_logger().error(
                "NamedSMOTE.fit_resample: underlying SMOTE raised an exception.\n"
                f"{self.feature_names = }\n"
                f"X.dtypes={X.dtypes.to_dict()}\n"
                f"Exception: {e}\nTraceback:\n{traceback.format_exc()}"
            )
            raise

        # Rebuild DataFrame with original column names if needed
        if isinstance(X_res, np.ndarray) and hasattr(X, "columns"):
            # Parent's SMOTE returns array; rebuild DataFrame with same selected columns
            cols = list(X.columns)
            X_res = pd.DataFrame(X_res, columns=cols)

        X_res = self._ensure_dataframe(X_res, feature_names=self.feature_names)

        logger.debug(f"DBG:SMOTERECUR00 - EXIT {self.__class__.__name__}.fit_resample")

        return X_res, y_res

class NamedSMOTENC(BaseNamedSamplerMixin, SMOTENC):
    """
    A SMOTENC over-sampler that handles categorical features by name.

    This class extends the standard `imbalanced-learn` SMOTENC to accept
    categorical feature names directly, providing a robust solution for
    mixed numerical and categorical data within a pandas DataFrame. It maps
    the provided names to their corresponding column indices during
    the `fit_resample` process.
    """
    def __init__(self, 
                 feature_names: Optional[List[str]] = None,
                 categorical_feature_names: List[str] = [], 
                 **kwargs):
        """
        Initializes the NamedSMOTENC over-sampler.

        Parameters
        ----------
        categorical_feature_names : list of str
            A list of column names corresponding to the categorical features.
            This argument is mandatory and must not be empty.
        **kwargs : dict
            Additional keyword arguments to be passed to the parent `SMOTENC` class.

        Raises
        ------
        ValueError
            If `categorical_feature_names` is an empty list.
        """

        # Run base initializations
        BaseNamedSamplerMixin.__init__(self, **kwargs)
        SMOTENC.__init__(self, categorical_features = [], **kwargs)

        # Store provided feature names and categorical metadata
        self.feature_names = feature_names or []
        if not categorical_feature_names:
            raise ValueError("NamedSMOTENC requires non-empty categorical_feature_names.")
        self.categorical_feature_names = list(categorical_feature_names)
        self.categorical_features_indices: List[int] = []
        self._get_logger().debug(f"DBG:SMOTERECUR00 - ENTER {self.__class__.__name__}.fit_resample")

        if not self.categorical_features_indices:
            self._get_logger().warning("No categorical features remain after selection; falling back to plain SMOTE.")
            smote = NamedSMOTE(random_state=gv.RANDOM_STATE, feature_names=self.feature_names)
            self._get_logger().debug("DBG:SMOTERECUR01 - FALLing back to NamedSMOTE from NamedSMOTENC")    
            return cast(
                Tuple[pd.DataFrame, np.ndarray], 
                smote.fit_resample(X, y)
            )

        # If input is not a DataFrame, try to convert it to DataFrame
        if not isinstance(X, pd.DataFrame):
            self._get_logger().debug(f"NamedSMOTENC received {type(X)}, converting to DataFrame.")
            try:
                X = pd.DataFrame(X, columns=self.feature_names)
            except Exception as e:
                self._get_logger().error(f"Failed to convert input to DataFrame: {e}")
                raise TypeError(f"NamedSMOTENC expects a pandas DataFrame, got {type(X)}") from e

        if not hasattr(X, "columns"):
            raise TypeError("NamedSMOTENC expects a pandas DataFrame.")

        X = self._ensure_dataframe(X)
        X_df = X.copy()

        # Map names -> indices (do this BEFORE any "fallback" decision)
        self.categorical_features_indices = []
        for cname in self.categorical_feature_names:
            if cname not in X_df.columns:
                self._get_logger().debug(f"NamedSMOTENC: categorical '{cname}' not in X; skipping.")
                continue
            loc = X_df.columns.get_loc(cname)
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

        # If mapping came back empty -> fallback to numeric SMOTE wrapper
        if not self.categorical_features_indices:
            self._get_logger().warning(
                "No categorical features remain after selection; falling back to plain SMOTE."
            )
            smote = NamedSMOTE(random_state=self.random_state, categorical_feature_names=self.categorical_feature_names)
            return cast(
                Tuple[pd.DataFrame, np.ndarray], 
                smote.fit_resample(X, y)
            )

        # Tell SMOTENC which indices are categorical
        self.categorical_features = self.categorical_features_indices
        
        # Convert to arrays for parent SMOTENC but keep X for column names
        X_arr = np.asarray(X) 
        y_arr = np.asarray(y)
        logging.debug(f"{type(X_arr) = }, {type(y_arr) = }")
        logging.debug(f"{type(X) = }, {type(y) = }")
        X_res, y_res = None, None
        try:
            result = super()._fit_resample(X_arr, y_arr)
            if len(result) == 2:
                X_res, y_res = result
            else:
                raise ValueError("Unexpected return value from fit_resample.")
        except Exception as e:
            # Log full context and traceback
            self._get_logger().error(
                "NamedSMOTENC.fit_resample: underlying SMOTENC raised an exception.\n"
                f"categorical_feature_names={self.categorical_feature_names}\n"
                f"mapped_indices={self.categorical_features_indices}\n"
                f"X.dtypes={X.dtypes.to_dict()}\n"
                f"Exception: {e}\nTraceback:\n{traceback.format_exc()}"
            )
            raise

        # Rebuild DataFrame with original column names if needed
        if isinstance(X_res, np.ndarray) and hasattr(X, "columns"):
            cols = list(X.columns)
            X_res = pd.DataFrame(X_res, columns=cols)

        X_res = self._ensure_dataframe(X_res)

        logger.debug(f"DBG:SMOTERECUR00 - EXIT {self.__class__.__name__}.fit_resample")

        return X_res, y_res
