"""
Pipeline debugging utilities for detecting data corruption and validation issues.
"""
import json
import pandas as pd
import numpy as np
import logging
import inspect
import sys
import traceback
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
import pandas.api.types as ptypes

# local imports (keep these as you had them)
from src.utils import gv, setup_logging

# CONSTANTS

# Configure logger
setup_logging(gv.LOG_DIR / f"{__name__.split('.')[-1]}.log")
logger = logging.getLogger(__name__)

# Attempt to import psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logger.warning("Warning: psutil not installed. Memory tracking will be disabled.")

# Helper predicates (use pandas API to avoid passing pandas ExtensionDtypes directly to np.issubdtype)
def is_numeric_dtype(dtype) -> bool:
    return ptypes.is_numeric_dtype(dtype)


def is_integer_or_bool_dtype(dtype) -> bool:
    return ptypes.is_integer_dtype(dtype) or ptypes.is_bool_dtype(dtype)


# Module-level "allowed" python types for category checking (used only for isinstance checks)
VALID_NUMERIC_BOOL_TYPES = (int, np.integer, bool)


class DataValidator:
    """
    Validates DataFrame integrity and detects corruption.

    This class performs a suite of checks on a DataFrame at various pipeline stages
    (e.g., input, output, transformation steps) to ensure data quality.

    Example:
        >>> validator = DataValidator(
        ...     name="PreprocessValidator",
        ...     categorical_feature_names=['cat_a'], 
        ...     ohe_column_names=['ohe_b_val1', 'ohe_b_val2']
        ... )
        >>> # Assuming df is a DataFrame
        >>> # report = validator.validate_frame(df, stage="initial_load")
    """

    def __init__(
        self,
        name: str,
        X: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        categorical_feature_names: Optional[List[str]] = None,
        ohe_column_names: Optional[List[str]] = None
    ):
        """
        Initialize the DataValidator.

        Args:
            name: A unique name for the validator instance.
            X: The initial feature DataFrame (optional, for reference).
            y: The initial target array (optional, for reference).
            categorical_feature_names: List of expected categorical columns.
            ohe_column_names: List of expected one-hot encoded columns.
        """
        self.name = name
        self.checks_performed: List[Dict[str, Any]] = []
        self.X = X if X is not None else pd.DataFrame()
        self.y = y if y is not None else np.array([])
        self.categorical_feature_names = categorical_feature_names or []
        self.ohe_column_names = ohe_column_names or []

    def validate_frame(self, df: pd.DataFrame, stage: str = "unknown") -> Dict[str, Any]:
        """
        Comprehensive DataFrame validation with detailed reporting.

        Performs checks for:
        1. Empty DataFrame.
        2. Missing expected feature columns (categorical/OHE).
        3. NaN/Inf values in any column (compacted loop).
        4. X/y length mismatch (for sampling stages like SMOTE).
        5. Suspicious dtypes in categorical columns.
        6. Non-binary values in OHE columns.
        7. Duplicate columns.

        Args:
            df: The DataFrame to validate.
            stage: The pipeline stage (e.g., 'input', 'SMOTE', 'transform')
                   for logging and reporting context.
        
        Returns:
            Dict with validation results and any issues found.

        Raises:
            ValueError: If critical errors are found AND DEBUG_MODE is True.
        """
        report: Dict[str, Any] = {
            "stage": stage,
            "validator": self.name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "issues": [],
            "warnings": [],
            "errors": [],
            "passed": True
        }

        # 1. Check for empty dataframe
        if df.empty:
            report["errors"].append(f"Input DataFrame is empty (shape={df.shape}).")

        # 2a. Check for missing categorical columns
        if self.categorical_feature_names:
            missing_categorical_cols = [
                col for col in self.categorical_feature_names if col not in df.columns
            ]
            if missing_categorical_cols:
                report["errors"].append(f"Missing categorical columns: {missing_categorical_cols}")
                logger.critical(f"Missing categorical features - missing_categorical_cols={missing_categorical_cols}")

        # 2b. Check for missing OHE columns
        if self.ohe_column_names:
            missing_ohe_cols = [col for col in self.ohe_column_names if col not in df.columns]
            if missing_ohe_cols:
                report["errors"].append(f"Missing One-hot Encoded columns: {missing_ohe_cols}")

        # 3. Check for NaN/Inf values
        bad_values = {}
        for col in df.columns:
            col_data = df[col]

            # Check NaN
            nan_count = int(col_data.isna().sum())
            if nan_count > 0:
                # store a sample of unique values (limit to reasonable size)
                uniques_sample = pd.unique(col_data.dropna())[:20]
                bad_values.setdefault("nan", []).append((col, nan_count, uniques_sample.tolist()))

            # Check Inf (only for numeric types)
            if is_numeric_dtype(col_data.dtype):
                # use to_numpy for isinf checks to avoid dtype problems
                arr = col_data.to_numpy()
                inf_mask = np.isinf(arr)
                inf_count = int(np.count_nonzero(inf_mask))
                if inf_count > 0:
                    uniques_sample = pd.unique(arr[~np.isnan(arr)][:20])
                    bad_values.setdefault("inf", []).append((col, inf_count, uniques_sample.tolist()))

        if bad_values.get("nan"):
            for col, count, uniques in bad_values["nan"]:
                report["errors"].append(f"Column '{col}' has {count} NaN values, uniques={uniques!r}")

        if bad_values.get("inf"):
            for col, count, uniques in bad_values["inf"]:
                report["errors"].append(f"Column '{col}' has {count} infinite values, uniques={uniques!r}")

        # 4. Check for X/y length mismatch (for SMOTE stages)
        if stage in ["SMOTE"] and self.y is not None and self.X is not None:
            try:
                 if len(self.y) != len(self.X):
                    report["errors"].append(f"X and y length mismatch: {len(self.X)} vs {len(self.y)} for SMOTE")
            except Exception:
                # defensive: if lengths cannot be determined, raise an error
                report["errors"].append("Unable to determine lengths of X and y for SMOTE check.")

        # 5. Check for unexpected dtypes in categorical columns
        valid_categorical_cols = [col for col in self.categorical_feature_names if col in df.columns]
        VALID_NUMERIC_BOOL_TYPES = (int, np.integer, bool) 

        for col in valid_categorical_cols:
            np_dtype = df[col].dtype

            # If it's numeric but not integer/bool, warn
            if not is_integer_or_bool_dtype(np_dtype):
                unique_python_types = {type(x) for x in df[col].dropna().values} # a set
                report["warnings"].append(
                    f"Categorical column '{col}' has NumPy dtype {np_dtype} and contains element types: {unique_python_types}"
                )

                # Find rows where a Python type is suspicious
                suspicious_mask = df[col].apply(lambda x: not isinstance(x, VALID_NUMERIC_BOOL_TYPES) and not pd.isna(x))
                suspicious_rows = df[suspicious_mask]
                if not suspicious_rows.empty:
                    report["warnings"].append(
                        f"Suspicious rows (non-integer/bool type) in column '{col}':\n"
                        f"{suspicious_rows}"
                    )

        # 6. Check for impossible values in one-hot encoded columns
        for col in self.ohe_column_names:
            if col in df.columns:
                unique_vals = pd.unique(df[col].dropna())
                # allow 0/1 only; also allow booleans which will show up as True/False
                allowed = {0, 1, True, False}
                if not set(unique_vals).issubset(allowed):
                    report["issues"].append(
                        f"Binary (One-hot Encoded) column '{col}' has non-binary values: {unique_vals!r}" #!r - output the more formal __repr__ representation instead of the readable __str__
                    )

        # 7. Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if df.columns.tolist().count(col) > 1]
            report["issues"].append(f"Duplicate columns found: {set(duplicates)}")

        # Set overall pass/fail
        report["passed"] = not any(report[key] for key in ["issues", "warnings", "errors"])

        # Log results


        ## Warnings and Issues
        for level, messages in [("warning", report["warnings"]), ("error", report["issues"])]:
            for msg in messages:
                getattr(logger, level)(f"  - {msg}")

        ## Log errors and raise if in DEBUG_MODE
        if report["errors"]:
            logger.critical(f"{self.name} - {stage}: {len(report['errors'])} critical errors found")
            for error in report["errors"]:
                logger.critical(f"  - {error}")

            if gv.DEBUG_MODE:
                msg = "\n  - ".join(report["errors"])
                raise ValueError (f"Significant errors with the data at {self.name}.{stage}, including:\n  - {msg}")

        self.checks_performed.append(report)
        return report

    def compare_frames(self, before: pd.DataFrame, after: pd.DataFrame, 
                      operation: str = "unknown", corr_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Compare two DataFrames (e.g., before/after a transform) to detect changes/corruption.

        Args:
            before: DataFrame before the operation.
            after: DataFrame after the operation.
            operation: Name of the operation/step.
            corr_threshold: Correlation threshold for multicollinearity check.
        
        Returns:
            A dictionary summarizing the comparison results and issues found.
        """
        comparison: Dict[str, Any] = {
            "operation": operation,
            "shape_before": before.shape,
            "shape_after": after.shape,
            "columns_added": [],
            "columns_removed": [],
            "dtype_changes": [],
            "value_corruption": [],
            "multicollinearity": [],
            "infos": [], # Informational only
            "warnings": [], # Warning only
            "issues": [], # Error only
            "errors": [], # Critical and raise flag/stop & scream
            "passed": False
        }

        # Column changes
        cols_before = set(before.columns)
        cols_after = set(after.columns)
        comparison["columns_added"] = list(cols_after - cols_before)
        comparison["columns_removed"] = list(cols_before - cols_after)
        if comparison["columns_added"]:
            comparison["infos"].append(f"New columns introduced: {list(cols_after - cols_before)}")
        if comparison["columns_removed"]:
            comparison["infos"].append(f"Old columns removed: {list(cols_before - cols_after)}")

        # Check common columns for dtype changes and value corruption
        common_cols = list(cols_before & cols_after)
        for col in common_cols:
            before_dtype = before[col].dtype
            after_dtype = after[col].dtype
            if before_dtype != after_dtype:
                comparison["dtype_changes"].append(
                    {"column": col, "before": str(before_dtype), "after": str(after_dtype)}
                )
                if is_integer_or_bool_dtype(before_dtype) and not is_integer_or_bool_dtype(after_dtype):
                    comparison["errors"].append(
                        f"Column '{col}' saw dtype changes from '{str(before_dtype)}' to '{str(after_dtype)}'"
                    )
                else:
                     comparison["warnings"].append(f"Column '{col}' saw dtype changes from '{str(before[col].dtype)}' to '{str(after[col].dtype)}'")

            # Check for new NaN/inf values (Refined to check for numeric types)
            if is_numeric_dtype(before[col].dtype) and is_numeric_dtype(after[col].dtype):
                before_arr = before[col].to_numpy()
                after_arr = after[col].to_numpy()
                before_bad = int(np.count_nonzero(~np.isfinite(before_arr)))
                after_bad = int(np.count_nonzero(~np.isfinite(after_arr)))

                if after_bad > before_bad:
                    corruption_count = after_bad - before_bad
                    comparison["value_corruption"].append(
                        {
                            "column": col,
                            "corruption_type": "NaN/inf introduced",
                            "before_count": before_bad,
                            "after_count": after_bad,
                        }
                    )
                    comparison["errors"].append(f"Column '{col}' gained {corruption_count} corrupt values (NaN/Inf)")

        # Multicollinearity Check
        if len(common_cols) > 1:
            try:
                # Use only numeric columns that are common between frames
                numeric_cols = [c for c in common_cols if is_numeric_dtype(after[c].dtype)]

                if len(numeric_cols) > 1:
                    # Fill for correlation calculation robustness
                    numeric_df: pd.DataFrame = after[numeric_cols].fillna(0).copy() # Fill for correlation calculation robustness
                    corr_matrix = numeric_df.corr().abs()

                    # Find high correlations (off-diagonal elements)
                    upper = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )

                    stacked = upper.stack()

                    # Find high correlation pairs.
                    high_corr_pairs = {}
                    for idx, val in stacked.items():
                        if bool(val >= corr_threshold):
                            high_corr_pairs[idx] = val
                    
                    for (col_a, col_b), corr_val in high_corr_pairs.items():
                        issue_msg = f"High correlation ({corr_val:.4f}) between '{col_a}' and '{col_b}'"
                        comparison["multicollinearity"].append({
                            "pair": (col_a, col_b),
                            "correlation": corr_val
                        })
                        comparison["issues"].append(issue_msg)

            except Exception as e:
                logger.error(f"Multicollinearity check failed for {operation}: {e}")
                comparison["issues"].append(f"Multicollinearity check failed: {e}")

        # Logging only for info, issues, and warnings
        for level, messages in [ ("info", comparison["infos"]), ("warning", comparison["warnings"]), ("error", comparison["issues"])]:
            for msg in messages:
                getattr(logger, level)(f"  - {msg}")

        # Logging and raise flag for errors
        for level, messages in [("critical", comparison["errors"])]:
            for msg in messages:
                getattr(logger, level)(f"  - {msg}")
        if comparison["errors"]:
            log = "\n - ".join(comparison["errors"])
            raise ValueError(f"Unacceptable transformations to DataFrame during {operation}: \n - {log}")

        comparison["passed"] = not any(comparison[key] for key in ["warnings", "issues", "errors"])
        return comparison


def debug_pipeline_step(step_name: str, validator_name: str = ""):
    """
    Decorator to wrap pipeline steps with data validation.
    
    This decorator performs input/output validation and frame comparison for
    pipeline methods (like fit, transform, fit_resample) that take (X, y)
    and return (X_out, y_out).

    Usage:
        class MyTransformer(BaseEstimator, TransformerMixin):
            # ... __init__ ...
            
            @debug_pipeline_step("FeatureScale")
            def transform(self, X):
                # your code
                return X_transformed
    
    Args:
        step_name: A human-readable name for the pipeline step (e.g., "SMOTE").
        validator_name: Optional name for the DataValidator instance.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Starting {step_name = }")
            except Exception:
                pass
            X_input, y_input, X_output, y_output, validator = None, None, None, None, None

            # 1. Validator Setup: Try to get or create a DataValidator instance
            instance = args[0] if args and hasattr(args[0], '__class__') else None
            class_name = instance.__class__.__name__ if instance else "Function"

            # Check if instance already has a validator; only create new validator if needed and if we have the required data
            if instance and hasattr(instance, '_validator'):
                validator = instance._validator
            elif instance and hasattr(instance, 'X') and hasattr(instance, 'y'):
                # Pass *only* the initial/reference data/features to the validator
                validator = DataValidator(
                    validator_name or f"{class_name}_{step_name}",
                    instance.X,
                    instance.y,
                    getattr(instance, 'categorical_feature_names', None),
                    getattr(instance, 'ohe_column_names', None)
                )

            # Find the input DataFrame and target array in the arguments
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    X_input = arg
                    # Check for y (assuming y follows X immediately in common methods)
                    if i + 1 < len(args) and isinstance(args[i + 1], (np.ndarray, pd.Series)):
                        y_input = args[i + 1]
                    break

            # Start profiling and validation
            start_time = pd.Timestamp.now()
            # profiler may be defined later as a global; access it at runtime
            try:
                profiler.track_memory("start", step_name)
            except Exception:
                # defensive: in case profiler not available yet
                logger.debug("Profiler not ready for track_memory('start').")

            # 2. Validate input
            if X_input is not None:
                logger.debug(f"=== {step_name} ENTRY ===")
                if validator:
                    input_report = validator.validate_frame(X_input, f"{step_name}_input")
                else:
                    input_report = {"passed": True}

                if not input_report.get("passed", True):
                    logger.error(f"{step_name} received corrupted input data!")

            # 3. Call the actual function
            try:
                result = func(*args, **kwargs)
                status = "success"
            except Exception as e:
                status = "failure"
                logger.error(f"{step_name} failed with error: {e}")
                if X_input is not None:
                    logger.error(f"Input data shape: {getattr(X_input, 'shape', None)}")
                    logger.error(f"Input data dtypes: {getattr(X_input, 'dtypes', None)}")
                try:
                    profiler.track_memory("end", step_name)
                except Exception:
                    pass
                raise

            # 4. Validate and compare output
            if isinstance(result, tuple) and len(result) == 2:
                X_output, y_output = result
            elif isinstance(result, pd.DataFrame):
                X_output = result

            if X_output is not None and isinstance(X_output, pd.DataFrame):
                logger.debug(f"=== {step_name} EXIT ===")

                if validator:
                    output_report = validator.validate_frame(X_output, f"{step_name}_output")

                if X_input is not None and validator:
                    comparison = validator.compare_frames(X_input, X_output, operation=step_name)
                    if not comparison.get("passed", True) and gv.DEBUG_MODE:
                        raise ValueError(f"{step_name} corrupted the data!")

            # 5. Log execution and save snapshot
            duration = (pd.Timestamp.now() - start_time).total_seconds()

            try:
                profiler.log_step(
                    step_name,
                    X_input.shape if X_input is not None else (0, 0),
                    X_output.shape if X_output is not None else (0, 0),
                    duration,
                status
                )
                profiler.track_memory("end", step_name)
            except Exception:
                logger.debug("Profiler not ready for log_step/track_memory('end').")

            if X_output is not None and y_output is not None:
                try:
                    profiler.save_snapshot(step_name, X_output, y_output)
                except Exception:
                    logger.debug("Profiler not ready for save_snapshot.")

            return result

        return wrapper

    return decorator


class PipelineProfiler:
    """
    Profile pipeline execution, track data flow, and monitor memory usage.
    (Memory tracking requires the optional psutil library).
    """
    def __init__(self):
        self.execution_log: List[Dict[str, Any]] = []
        self.data_snapshots: Dict[str, Any] = {}
        self.memory_log: Dict[str, Any] = {}

    def track_memory(self, event: str, step_name: str):
        """
        Record the current process memory usage.
        
        Args:
            event: 'start' or 'end' of a step.
            step_name: The name of the pipeline step.
        """
        if not PSUTIL_AVAILABLE or psutil is None:
            return

        try:
            process = psutil.Process()  # Now Pylance knows psutil is not None
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)

            key = f"{step_name}_{event}"
            self.memory_log[key] = {
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "timestamp": pd.Timestamp.now()
            }
            logger.debug(f"Memory {event} for '{step_name}': RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")
        except Exception as e:
            logger.warning(f"Failed to track memory for {step_name}: {e}")

    def log_step(self, step_name: str, input_shape: Tuple, output_shape: Tuple, 
                duration: float, status: str = "success"):
        """Log execution of a pipeline step."""
        entry: Dict[str, Any] = {
            "step": step_name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "duration_ms": duration * 1000,
            "status": status,
            "timestamp": pd.Timestamp.now(),
        }

        # Integrate memory usage into the step log if available
        start_mem = self.memory_log.get(f"{step_name}_start")
        end_mem = self.memory_log.get(f"{step_name}_end")

        if start_mem and end_mem:
            entry["memory_start_mb"] = start_mem["rss_mb"]
            entry["memory_end_mb"] = end_mem["rss_mb"]
            entry["memory_delta_mb"] = end_mem["rss_mb"] - start_mem["rss_mb"]

        self.execution_log.append(entry)
        logger.info(f"Pipeline step '{step_name}': {input_shape} -> {output_shape} ({duration:.3f}s)")

    def save_snapshot(self, step_name: str, X: pd.DataFrame, y: np.ndarray):
        """Save data snapshot for debugging."""
        self.data_snapshots[step_name] = {
            "X_shape": X.shape,
            "X_dtypes": {c: str(t) for c, t in X.dtypes.items()},
            "X_sample": X.head(3).to_dict(),
            "y_shape": getattr(y, "shape", None),
            "timestamp": pd.Timestamp.now()
        }

    def get_summary(self) -> pd.DataFrame:
        """Get execution summary as DataFrame."""
        return pd.DataFrame(self.execution_log)

    def export_debug_info(self, filepath: Path):
        """Export all debug information to file."""
        debug_info = {
            "execution_log": self.execution_log,
            "data_snapshots": self.data_snapshots,
            "memory_log": self.memory_log,
            "export_timestamp": pd.Timestamp.now().isoformat()
        }


        with open(filepath, "w") as f:
            json.dump(debug_info, f, indent=2, default=str)

        logger.info(f"Debug information exported to {filepath}")

# Global profiler instance
profiler = PipelineProfiler()

def get_caller_info() -> str:
    """Get information about the calling function for debug logs."""

    # Start at the current frame (Frame 0: get_caller_info)
    current_frame = inspect.currentframe()

    # Move back one frame (Frame 1: The function that called get_caller_info)
    caller_frame = current_frame.f_back if current_frame else None

    # Move back a second frame (Frame 2: The original caller we want to log)
    # This is the frame containing the log statement itself.
    target_frame = caller_frame.f_back if caller_frame else None

    if target_frame is None:
        return "UNKNOWN_CALLER:0"  # Return a safe fallback string

    # Use 'target_frame' which is now guaranteed by the check to be a FrameType object
    filename = target_frame.f_code.co_filename
    line_number = target_frame.f_lineno
    function_name = target_frame.f_code.co_name

    return f"{Path(filename).name}:{function_name}:{line_number}"


### Specific debug functions
# --------------------------
# Debug parameters/functions/replacements
# --------------------------
sys.setrecursionlimit(1000)  # artificially low to fail faster, typically ~1000. Dangerous to exceed 2000-3000

def debug_check_for_nans(X: pd.DataFrame, categorical_features: list[str]) -> None:
    for col in categorical_features:
        if col in X.columns:
            bad_mask = ~np.isfinite(pd.to_numeric(X[col], errors="coerce"))
            if bad_mask.any():
                bad_rows = X[bad_mask]
                print(f"\n[DEBUG] Found non-finite in column '{col}':")
                print(bad_rows[[col]].head(20))  # show first 20 offending rows
                # Also log row indices for traceability
                print(f"[DEBUG] Row indices with issue: {list(bad_rows.index)}\n")


def debug_check_frame(df: pd.DataFrame, name: str):
    """Check DataFrame for NaN and infinite values."""
    # Check for NaN in any column
    nan_mask = df.isna().any(axis=1)
    
    # Check for inf in numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        inf_array = (~np.isfinite(numeric_df)).any(axis=1)
        inf_mask: pd.Series = pd.Series(inf_array, index=df.index)
    else:
        inf_mask = pd.Series(False, index=df.index)

    
    # Combine masks (both are now Series with same index)
    bad_mask = nan_mask | inf_mask
    
    logger.debug(f"DEBUG checking name={name}: rows={df.shape[0]}, cols={df.shape[1]}")
    
    if bad_mask.any():
        print(f"\n[DEBUG] {name} has NaNs/Infs in {int(bad_mask.sum())} rows")
        
        # Find problematic columns
        nan_cols = df.columns[df.isna().any()].tolist()
        
        # Check for inf in numeric columns
        inf_cols = []
        if not numeric_df.empty:
            inf_cols = numeric_df.columns[
                numeric_df.isin([np.inf, -np.inf]).any()
            ].tolist()
        
        problem_cols = set(nan_cols + inf_cols)
        
        for col in problem_cols:
            bad_vals = df.loc[bad_mask, col].unique()
            print(f"  Column {col}: {bad_vals}")

def run_cv_debug(pipeline, X, y, cv, scoring="f1"):
    """Manual cross-validation with debug logging for bad values."""
    fold_scores = []

    # Determine if X and y are pandas objects to use appropriate indexing
    is_pandas_X = isinstance(X, pd.DataFrame)
    is_pandas_y = isinstance(y, (pd.Series, pd.DataFrame)) # np.ndarray is also acceptable for y for consistency

    logger.debug(f"[DEBUG] My_cv_debug: {is_pandas_X = } {is_pandas_y = }")

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if is_pandas_X:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]

        if is_pandas_y:
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]

        # Check for NaN/inf in the offending column
        if is_pandas_X and "AutomaticFinancing_below_600_" in X_train.columns:
            bad = X_train.loc[X_train["AutomaticFinancing_below_600_"].isin([np.nan, np.inf, -np.inf])]
            if not bad.empty:
                logger.error(f"[DEBUG] BAD VALUES in fold {fold_i} of 'AutomaticFinancing_below_600_'")
                logger.error(bad.head())

    return {"test_score": np.array(fold_scores), "mean_score": np.mean(fold_scores) if fold_scores else float("nan")}


# Debug Transformer to inspect data at various pipeline stages
class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tag="debug"):
        self.tag = tag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\n[DEBUG {self.tag}] Transform called on data with shape: {getattr(X, 'shape', None)}")
        if isinstance(X, pd.DataFrame) and "AutomaticFinancing_below_600_" in X.columns:
            col = X["AutomaticFinancing_below_600_"]
            bad_mask = col.isna() | np.isinf(col)
            if bad_mask.any():
                print(f"\n[DEBUG {self.tag}] BAD values detected in 'AutomaticFinancing_below_600_':")
                print(col[bad_mask].head())
            else:
                print(f"\n[DEBUG {self.tag}] All values good. uniques={col.unique()!r}")
        return X

# --- Monkey Patch Debugging for ImbSMOTE.fit_resample ---
# Temporarily replace ImbSMOTE.fit_resample with a debug version that logs entry and exit
if False:
    real_fit_resample = ImbSMOTE.fit_resample

    def debug_fit_resample(self, X, y, **params):
        print(f"ENTER Monkey SMOTE.fit_resample id={id(self)} class={self.__class__}")
        return real_fit_resample(self, X, y, **params)

    ImbSMOTE.fit_resample = debug_fit_resample
else:
    logger.debug("No monkey business")
    pass