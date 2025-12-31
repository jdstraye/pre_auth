'''
Contains utility functions for eval_algos.py
Tests in 
  - tests/test_parameter_sampling.py
'''
from typing import Any, Dict, List
import logging
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterSampler
from pathlib import Path
from src.utils import gv
import os
import shutil
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# --------------------------
# Helpers - checks, files, time, symlink
# --------------------------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def log_class_imbalance(y: Union[np.ndarray, Sequence[int]], phase: str) -> None:
    """Logs the class distribution and warns about potential issues with rare classes."""
    counts = pd.Series(list(y)).value_counts().sort_index()
    logger.info(f"{phase} class imbalance: {counts.to_dict()}")

    # Warn if any class has fewer than 5 samples, which is often problematic for SMOTE/KFold
    if any(count < 5 for count in counts):
        logger.warning(f"{phase}: Some classes have fewer than 5 samples, which may cause issues with StratifiedKFold or SMOTENC.")
        for label, count in counts.items():
            logger.info(f"  Class {label}: {count} samples")

def calc_n_iter_model(param_grid: Mapping[str, Iterable[Any]]) -> int:
    """
    Calculate the number of iterations for ParameterSampler based on the parameter grid.
    It should be the sum of the lengths of each parameter's value list.
    Example:
    dist = {
        'param1': [1, 2, 3],
        'param2': ['a', 'b', 'c', 'd'],
        'param3': 5
    }
    param1 has 3 options, param2 has 4 options, param3 has 1 option (not a list), so the total is 8.


    Args:
        param_grid (Dict[str, Iterable[Any]]): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        int: Calculated number of iterations, at least 1 for every parameter.

    Raises:
        ValueError: If param_grid is empty or contains empty value lists.
    """
    if not param_grid:
        raise ValueError("Parameter grid is empty.")
    lengths = [len(v) if isinstance(v, Sequence) else 1 for v in param_grid.values()]
    sum_lengths = sum(lengths)

    return sum_lengths

def extract_valid_tier_records(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to include only records that have a valid tier label. To be a valid tier, the status must be approved or refunded
    and the final_contract_tier_label must exist. For example, "rejected" are approved, but they do not receive a
    tier, so they must be excluded from the tier_df.
    A valid tier record must also have a non-null 'final_contract_tier_label' and
    it must not be one of the 'NA', '-1', '', or 'null' string representations.

    Args:
        in_df (pd.DataFrame): The input DataFrame containing all records.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only records with valid tier information.
    """
    # Setup
    tier_target_col = 'final_contract_tier_label'
    logger.info(f"Filtering for valid '{tier_target_col}' records.")

    # Ensure the target column exists
    if tier_target_col not in in_df.columns:
        logger.error(f"Target column '{tier_target_col}' not found in DataFrame.")
        columns_str = '\n - '.join(in_df.columns)
        raise ValueError(f"Target column '{tier_target_col}' not found in DataFrame columns: {columns_str}")

    # Convert to string for consistent comparison and handling of NaNs
    tier_str = in_df[tier_target_col].astype(str).str.lower() # Use .lower() for case-insensitivity

    # Define invalid tier string representations
    invalid_tiers = ["na", "-1", "", "null", "none"]

    # Filter out records where the tier target is null or among the invalid strings
    valid_tier_df = in_df[
        (~tier_str.isin(invalid_tiers)) &
        (in_df[tier_target_col].notnull()) # This catches actual NaNs/None that astype(str) might turn to 'nan'
    ]

    logger.info(f"Extracted {len(valid_tier_df)} valid tier records out of {len(in_df)} total.")
    return valid_tier_df


# --------------------------
# Save report and utility (re-usable)
# --------------------------
def save_report(y_true, y_pred, prefix: str, report_dir: Path) -> Dict[str, Any]:
    """
    Saves a classification report to a CSV file in report_dir and returns it as a dictionary.

    Args:
        y_true (Sequence): True labels.
        y_pred (Sequence): Predicted labels.
        prefix (str): Prefix for the report filename.
        report_dir (Path, optional): Directory to save the report. Defaults to LOG_DIR.

    Returns:
        Dict[str, Any]: Classification report as a dictionary.

    Raises:
        TypeError: If y_true or y_pred is not a sequence, or if report_dir is not a Path.
        ValueError: If prefix is empty, or if classification_report does not return a dictionary.
        OSError: If the report cannot be saved to the specified directory.
    """
    # Validate inputs
    valid_types = (np.ndarray, list, tuple, pd.Series)


    if not isinstance(y_true, valid_types) and not (isinstance(y_true, Sequence) and not isinstance(y_true, (str, bytes))):
        raise TypeError(f"y_true must be array-like, got {type(y_true)}")

    if not isinstance(y_pred, valid_types) and not (isinstance(y_pred, Sequence) and not isinstance(y_pred, (str, bytes))):
        raise TypeError(f"y_pred must be array-like, got {type(y_pred)}")

    if not isinstance(prefix, str) or not prefix:
        logger.error("prefix must be a non-empty string.")
        raise ValueError("prefix must be a non-empty string.")
    if not isinstance(report_dir, Path):
        logger.error("report_dir must be a Path object.")
        raise TypeError("report_dir must be a Path object.")

    # Generate classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Validate report_dict
    if not isinstance(report_dict, dict):
        logger.error(f"Expected classification_report to return a dict, got {type(report_dict)}.")
        raise ValueError(f"Expected classification_report to return a dict, got {type(report_dict)}.")

    try:
        df_report = pd.DataFrame(report_dict).transpose()
        report_file = gv.LOG_DIR / f"{prefix}_report.csv"
        df_report.to_csv(report_file)
        logger.info(f"Saved '{prefix}' classification report to '{report_file}'")
    except OSError as e:
        logger.error(f"Failed to save classification report to '{report_file}': {str(e)}")
        raise OSError(f"Failed to save classification report to '{report_file}': {str(e)}") from e

    return report_dict

def generate_parameter_samples(param_dist: Dict[str, List[Any]],
                               n_samples: float,
                               random_state: int
    ) -> List[Dict[str, Any]]:
    """
    Use sklearn.model_selection.ParameterSampler to create a list of candidate parameter dicts,
    but applies certain constraints, namely:
     - Mutual exclusion of certain parameters, feature_selecting_classifier__threshold and 
       feature_selecting_classifier__max_features. They will have equal counts in the final samples.

    Args:
      param_dist: mapping param -> list of possible values (each must be an iterable/list)
      n_samples: target number of valid samples to return
      random_state: seed

    Returns:
      list of param dicts (length <= n_samples if not enough unique valid combos)
    """
    # Split n_samples in half
    n_half = math.ceil(n_samples / 2)

    # Half the samples will have max_features set, threshold=None
    param_dist_maxf = {
        k: v for k, v in param_dist.items()
        if k != "feature_selecting_classifier__threshold"
    }
    param_dist_maxf["feature_selecting_classifier__threshold"] = [None]

    # Half the samples will have threshold set, max_features=None
    param_dist_thresh = {
        k: v for k, v in param_dist.items()
        if k != "feature_selecting_classifier__max_features"
    }
    param_dist_thresh["feature_selecting_classifier__max_features"] = [None]

    # Generate samples for each sub-grid
    sampler_maxf = ParameterSampler(param_dist_maxf, n_iter=n_half, random_state=random_state)
    sampler_thresh = ParameterSampler(param_dist_thresh, n_iter=n_half, random_state=random_state)

    # Combine and shuffle
    samples = list(sampler_maxf) + list(sampler_thresh)
    np.random.shuffle(samples)

    # Filter incompatible combinations (e.g., OHE encoding incompatible with SMOTENC)
    def is_valid_candidate(candidate: Dict[str, Any]) -> bool:
        enc = candidate.get('encoding', None)
        smm = candidate.get('smote__method', None)
        # OHE cannot be used with SMOTENC (which requires ordinal encoding)
        if enc == 'ohe' and smm == 'smotenc':
            return False
        return True

    filtered = [s for s in samples if is_valid_candidate(s)]
    if len(filtered) < n_samples:
        logger.info(f"Requested {n_samples} samples but only {len(filtered)} compatible samples available after filtering.")

    logger.info(f"Generated {len(filtered[:int(n_samples)])} valid parameter samples.")
    logger.debug(f"Parameter samples: {filtered[:int(n_samples)] = }")
    return filtered[:int(n_samples)]

def write_csv_and_symlink(df: pd.DataFrame, dest_dir: Path, basename: str, timestamp: str):
    """
    Write df to dest_dir / f"{basename}-{timestamp}.csv" and create/overwrite symlink basename-latest.csv.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{basename}-{timestamp}.csv"
    full = dest_dir / filename
    df.to_csv(full, index=False)
    # update symlink
    symlink = dest_dir / f"{basename}-latest.csv"
    try:
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(full.name)
    except Exception:
        # Some filesystems or permissions may not allow symlink; in that case, just overwrite the file
        logger.warning(f"Unable to create symlink {symlink} -> {full}; skipping symlink creation.")
    return full

def write_json_and_symlink(obj: Any, dest_dir: Path, basename: str, timestamp: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{basename}-{timestamp}.json"
    full = dest_dir / filename
    with open(full, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=str)
    symlink = dest_dir / f"{basename}-latest.json"
    try:
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(full.name)
    except Exception:
        logger.warning(f"Unable to create symlink {symlink} -> {full}; skipping symlink creation.")
    return full

