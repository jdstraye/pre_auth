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
    def _class_counts(self, y: Union[pd.Series, np.ndarray]):
        """
        Return a dictionary mapping class labels to their counts in y.
        Accepts a pandas Series or numpy array.
        """
        if isinstance(y, pd.Series):
            values = y.values
        else:
            values = np.asarray(y)
        # Ensure values is a numpy array of supported dtype for np.unique
        values = np.asarray(values)
        # Only call to_numpy if it's a pandas object
        if hasattr(values, 'to_numpy') and not isinstance(values, np.ndarray):
            values = values.to_numpy(dtype=None, copy=False)
        values = np.asarray(values)
        if values.dtype == object:
            values = values.astype(str)
        unique, counts = np.unique(values, return_counts=True)
        return dict(zip(unique, counts))

    def _minority_share(self, class_counts: Dict[Any, int]) -> float:
        """
        Return the share of the minority class given a class count dictionary.
        """
        if not class_counts:
            return 0.0
        min_count = min(class_counts.values())
        total = sum(class_counts.values())
        if total == 0:
            return 0.0
        return min_count / total

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
        return {
            "enabled": self.enabled,
            "categorical_feature_names": self.categorical_feature_names or None,
            "k_neighbors": self.k_neighbors,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
            "min_improvement": self.min_improvement
        }

    def set_params(self, **params):
        # No docstring to avoid syntax/indentation issues
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Accept 'smote__...' style if needed (pipeline will pass direct names)
                setattr(self, k, v)
        return self

    def _get_categorical_indices(self, X: pd.DataFrame) -> List[int]:
        indices = []
        for cat_name in self.categorical_feature_names:
            if cat_name in X.columns:
                idx = X.columns.get_loc(cat_name)
                indices.append(idx)
        return indices

    def _adjust_k_neighbors(self, y: Union[np.ndarray, pd.Series]) -> int:
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
        import numpy as np
        return np.asarray(y)

    @debug_pipeline_step("MaybeSMOTESampler-_fit_resample")
    def _fit_resample(self, X: Any, y: Any):
        # Save input types for output restoration
        X_type = type(X)
        y_type = type(y)
        X_index = getattr(X, 'index', None)
        X_columns = getattr(X, 'columns', None)
        y_index = getattr(y, 'index', None)
        X_is_sparse = sparse.issparse(X)

        # If disabled, pass-through with safe types
        if not self.enabled:
            self._get_logger().debug("MaybeSMOTESampler: disabled - passthrough")
            if X_is_sparse:
                X_out = X.copy()
            else:
                X_df = self._ensure_dataframe(X)
                if X_type is np.ndarray:
                    X_out = X_df.values
                elif X_type is list:
                    X_out = X_df.values.tolist()
                else:
                    X_out = X_df
            y_arr = np.asarray(y).ravel()
            if y_type is pd.Series:
                y_out = pd.Series(y_arr, index=y_index)
            elif y_type is list:
                y_out = y_arr.tolist()
            else:
                y_out = y_arr
            return X_out, y_out

        lg = self._get_logger()
        lg.debug(f"[LOGGER STATE] name={lg.name} effective={logging.getLevelName(lg.getEffectiveLevel())} handlers={len(lg.handlers)} propagate={lg.propagate}")
        lg_root = logging.getLogger()
        lg_root.debug(f"[ROOT HANDLERS] count={len(lg_root.handlers)}  root_level={logging.getLevelName(lg_root.level)}")

        # Always work with a DataFrame for internal processing
        X_df = self._ensure_dataframe(X)
        # Raise error if DataFrame is empty (for test compliance)
        if not X_is_sparse and X_df.empty:
            raise ValueError("Input DataFrame is empty")

        # Check for NaN values in categorical columns (for test compliance)
        if self.categorical_feature_names:
            for col in self.categorical_feature_names:
                if col in X_df.columns:
                    n_nans = X_df[col].isna().sum()
                    if n_nans > 0:
                        raise ValueError(f"Column '{col}' has {n_nans} NaN values")

        #20251004 # Flatten y to 1D before creating the pandas Series
        #20251004 y_flat = np.asarray(y).ravel()
        #20251004 y_series = pd.Series(y_flat)
        
        # Normalize y into a 1-D label vector
        arr_y = np.asarray(y)


        # Always set _y_format to avoid AttributeError
        self._y_format = None
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
                    self._y_format = "onehot"
                elif np.allclose(row_sums, 1) and not np.all((arr_y == 0) | (arr_y == 1)):
                    # Case B: probability vectors
                    self._get_logger().warning(
                        "y appears to be probability vectors; converting to 1-D via argmax(axis=1)."
                    )
                    y_flat = arr_y.argmax(axis=1)
                    self._y_format = None
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

        # If the user requested categorical columns that are entirely missing from X, we should
        # allow a graceful fallback (to NamedSMOTE / RandomOverSampler) rather than fail
        # during initial validation. Compute an effective categorical list that only contains
        # columns present in the frame.
        effective_categorical = [c for c in (self.categorical_feature_names or []) if c in X_df.columns]
        skip_validator = False
        if self.categorical_feature_names and not effective_categorical and self.allow_fallback:
            self._get_logger().warning(f"Missing requested categorical columns {self.categorical_feature_names}; will fall back to non-categorical sampler due to allow_fallback=True")
            skip_validator = True

        self._get_logger().debug(f"Frame Validation before SMOTE (skip_validator={skip_validator}) ...")
        if not skip_validator:
            self._validator.validate_frame(X_df, "SMOTE")
        else:
            self._get_logger().debug("Skipping DataValidator categorical checks due to allow_fallback condition")

        orig_counts = self._class_counts(y_series)
        orig_min_share = self._minority_share(orig_counts)

        # Adjust k_neighbors if too large
        self.k_neighbors = self._adjust_k_neighbors(y_flat)

        # Confirm X >< y correlation
        if X_df.shape[0] != len(y_series):
            raise ValueError("X and y length mismatch")

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
            # Use the effective categorical list computed earlier - this allows graceful fallback
            if effective_categorical:
                sampler = NamedSMOTENC(
                    feature_names=self.feature_names,
                    categorical_feature_names=effective_categorical,
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
        # Allow zero improvement if input is already perfectly balanced
        if self.min_improvement:
            if orig_min_share == new_min_share and orig_min_share == 0.5:
                # Already perfectly balanced, allow
                pass
            elif (new_min_share/orig_min_share)-1 < float(self.min_improvement):
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

        # Restore output type to match input
        if X_is_sparse:
            # Convert DataFrame back to sparse matrix
            X_out = sparse.csr_matrix(X_res.values)
        elif X_type is np.ndarray:
            X_out = X_res.values
        elif X_type is list:
            X_out = X_res.values.tolist()
        elif X_type is pd.Series:
            # Not typical for X, but handle for completeness
            X_out = pd.Series(X_res.values.flatten(), index=X_index)
        else:
            X_out = X_res
        if y_type is pd.Series:
            y_out = pd.Series(y_res, index=None if y_index is None else range(len(y_res)))
        elif y_type is list:
            y_out = y_res.tolist()
        else:
            y_out = y_res
        return X_out, y_out
    
    def get_feature_names_out(self, input_features=None):
        # Docstring removed to resolve indentation issues
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
        # Return tags as lists of str per sklearn API
        return {
            'stateless': [],
            'requires_y': [],
            'sampler': [],
        }

class NamedSMOTE(BaseNamedSamplerMixin, SMOTE):
    def __init__(self, feature_names: Optional[List[str]] = None, **kwargs):
        BaseNamedSamplerMixin.__init__(self, **kwargs)
        SMOTE.__init__(self, **kwargs)
        self.feature_names = feature_names or []

    def _fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        X = self._ensure_dataframe(X)
        X_df = X.copy()
        if not hasattr(X_df, "columns"):
            raise TypeError("NamedSMOTE expects a pandas DataFrame.")
        self._get_logger().debug(f"DBG:SMOTERECUR01 FEATURES -\n {self.feature_names = }\n {X.columns = }")
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        logging.debug(f"{type(X_arr) = }, {type(y_arr) = }")
        logging.debug(f"{type(X) = }, {type(y) = }")
        logger.debug(f"DBG:SMOTERECUR01 - pre-super {self.__class__.__name__}.fit_resample")
        logger.debug(f"DBG:SMOTERECUR01a - {self.__class__.__name__}\n{X_arr = }\n{y_arr = }")
        logger.debug(f"DBG:SMOTERECUR02 - post-super {self.__class__.__name__}.fit_resample")
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
    def __init__(self, 
                 feature_names: Optional[List[str]] = None,
                 categorical_feature_names: List[str] = [], 
                 **kwargs):

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

        # Do not attempt to perform any fallback here: we don't have access to X/y yet in __init__.
        # The mapping from categorical names -> indices and any fallback to NamedSMOTE is handled
        # within NamedSMOTENC._fit_resample where X and y are available.
        # Leaving this as a no-op prevents inadvertent UnboundLocalError caused by referencing X/y here.

    def _fit_resample(self, X: Any, y: Any) -> Tuple[Any, Any]:
        # Docstring removed to resolve unterminated string literal error
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
