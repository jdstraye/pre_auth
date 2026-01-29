"""
Helper functions common throughout the PreAuth ML model-making project.
"""
""" 
Creates a custom log handler that creates a fresh log file with a date in the name but also creates a symlink to the latest log file.
"""
import logging
import re
import pandas as pd
from typing import List, Tuple, Union, Sequence, Dict
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import os
import sys

# --------------------------
# Global Setup: Constants, Paths, & Logging
# --------------------------
class GlobalConfig:
    def __init__(self):
        # Constants
        ## Random Seed
        self.RANDOM_STATE = 42

        ## Number multiplier to define how many samples to try (heuristic, per-model)
        self.RANDOM_SEARCH_ITER_MULT = 0.1

        ## Defaults for SMOTE sanity check (minimum delta in minority share after resampling)
        self.DEFAULT_SMOTE_MIN_IMPROVEMENT = 0.01

        ## Logging Levels
        self.DEBUG_MODE = True  # Set to False to disable debug checks and critical error raising
        self.FILE_LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARN, ERROR, CRITICAL
        self.CONSOLE_LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARN, ERROR, CRITICAL
#        self.CONSOLE_LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, CRITICAL

        # Paths
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.REPORTS_DIR = self.LOG_DIR / "reports"
        self.PROBA_DIR = self.LOG_DIR / "proba"
        self.DEFAULT_SCHEMA_PATH = Path("src/column_headers.json")

        ## Path Setup
        for d in [self.LOG_DIR, self.MODELS_DIR, self.REPORTS_DIR, self.PROBA_DIR]:
            d.mkdir(parents=True, exist_ok=True)

gv = GlobalConfig()

# --- Custom Log Handler & Logging Setup (from previous code) ---
class CustomRotatingFileHandler(RotatingFileHandler):
    """
    A custom rotating file handler that appends a timestamp to the log file name.
    """
    def __init__(self, filename, maxBytes, backupCount):
        self.base_filename = Path(filename).stem
        self.base_dir = Path(filename).absolute().parent
        self.current_datetime = datetime.now().strftime("%Y%m%d.%H%M")
        self.baseFilename = str(self.base_dir / f"{self.base_filename}-{self.current_datetime}.log")
        super().__init__(self.baseFilename, maxBytes=maxBytes, backupCount=backupCount)
        self.latest_log_path = self.base_dir / f"{self.base_filename}_latest.log"
        if self.latest_log_path.exists() or self.latest_log_path.is_symlink():
            self.latest_log_path.unlink()
        self.latest_log_path.symlink_to(Path(self.baseFilename).relative_to(self.base_dir))

    def doRollover(self):
        """
        Modified rollover to handle the timestamped log file names correctly.
        """
        if self.stream:
            self.stream.close()
        current_datetime = datetime.now().strftime("%Y%m%d.%H%M")
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.{i}.log"
                dfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.{i + 1}.log"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = f"{self.base_dir}/{self.base_filename}-{current_datetime}.1.log"
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(self.baseFilename, dfn)
        self.baseFilename = f"{self.base_dir}/{self.base_filename}-{current_datetime}.log"
        self.stream = open(self.baseFilename, 'w')



# --- Logging ---
def debug_setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[


            logging.StreamHandler(),  # Log to console
        ]
    )

def setup_logging(log_file: Path):
    """
    Modifies the root logger to setup a rotating file handler and a console handler for logging. 
    It needs to be called once at the start of the program to setup 
    the file location and log levels.

    Args:
        log_file (Path): The path to the log file.

    After that initial call, it can be used by just calling
     logger = logging.getLogger(__name__)
    as usual to get a logger in that module.
    The file handler also creates/updates a symlink to the latest log file.
    The log levels for file and console are set in utils.py as part of the GlobalConfig class,
    FILE_LOG_LEVEL and CONSOLE_LOG_LEVEL.
    """
    
    print("DEBUG: setup_logging() was called!")  # <-- Add this

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clean up existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Prepare formatters
    file_formatter = logging.Formatter('%(filename)s:%(lineno)d - %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s')

    # Valid log levels
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # File handler
    ## Check if a file handler for this log_file already exists
    file_handler_exists = any(
        isinstance(h, CustomRotatingFileHandler) and h.baseFilename == str(log_file)
        for h in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = CustomRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_level = gv.FILE_LOG_LEVEL.upper()
        if file_level not in valid_levels:
            raise ValueError(f"Invalid file log level: {gv.FILE_LOG_LEVEL}. Must be one of {valid_levels}")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    ## Check if a console handler already exists
    console_handler = None
    console_handler_exists = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers
        #isinstance(type(h), logging.StreamHandler) for h in root_logger.handlers
    )
    print(f"DEBUG: {console_handler_exists = }, {root_logger.handlers = }")  # <-- Add this
    print("DEBUG: root_logger.handlers = [")
    for i, handler in enumerate(root_logger.handlers):
        print(f"  {i}: {type(handler).__name__} - {handler}")
    print("]")

    if not console_handler_exists:
        console_handler = logging.StreamHandler()
        console_level = gv.CONSOLE_LOG_LEVEL.upper()
        if console_level not in valid_levels:
            raise ValueError(f"Invalid console log level: {gv.CONSOLE_LOG_LEVEL}. Must be one of {valid_levels}")
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        print(f"DEBUG: Logging configured for: {log_file}")  # <-- Add this

# Picklable get_logger for every class/module
def get_logger(name: str) -> logging.Logger:
    """
    Centralized logger setup for all classes. Unlike setup_logging(), this can be 
    pickled because it does not use a FileHandler or store the Logger object on 
    the instance.
    Args:
        name: Logger name (e.g., f"{__name__}.{ClassName}").
    Returns:
        logging.Logger: Configured logger with handlers.
    """
    lg = logging.getLogger(name)
    if lg.level == logging.NOTSET:
        lg.setLevel(gv.CONSOLE_LOG_LEVEL.upper())
    lg.propagate = False

    # Attach handlers if missing
    if not lg.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        if ch.level == logging.NOTSET:
            ch.setLevel(gv.CONSOLE_LOG_LEVEL.upper())
        ch.setFormatter(logging.Formatter(
            "%(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
        ))
        lg.addHandler(ch)

    return lg

logger = logging.getLogger(__name__)

def load_column_headers(column_headers_json: Path, df: pd.DataFrame, classifier_type: str = None, allow_missing_columns: bool = False) -> Dict[str, List[str]]:
    """
    Loads feature, categorical, and target column names from a JSON schema file
    and validates that feature columns exist in the DataFrame.

    Args:
        column_headers_json (Path): Path to the JSON file containing column definitions.
        df (pd.DataFrame): The DataFrame to check for column existence.

    Returns:
        Dict:
            "feature_cols": List of sanitized names for feature columns ('X' == 'True').
            "categorical_cols": List of names for categorical columns ('categorical' == 'True').
            "target_cols": List of names for target columns ('Y' == 'True').
            "ohe_cols": List of names for columns that are one-hot encoded ('ohe_from' exists).

    Raises:
        FileNotFoundError: If the schema JSON file is not found.
        json.JSONDecodeError: If the schema file is not valid JSON.
        ValueError: If a required feature column from the schema is not found in the DataFrame.
    """
    logger.info(f"Loading column headers from: {column_headers_json}")
    try:
        with open(column_headers_json, 'r', encoding='utf-8') as f:
            header_data = json.load(f)
        # Normalize schema: remove duplicate column definitions while merging metadata
        # If duplicates exist, preserve the first occurrence but merge any missing keys from later entries
        seen = {}
        normalized = []
        for col in header_data:
            name = col['name']
            if name not in seen:
                seen[name] = dict(col)
                normalized.append(seen[name])
            else:
                # merge missing keys from duplicate into the canonical record
                existing = seen[name]
                for k, v in col.items():
                    if k not in existing or existing[k] in (None, '', 'False'):
                        existing[k] = v
        if len(normalized) != len(header_data):
            logger.info('Removed duplicate column definitions from schema; preserved first occurrence for each name and merged metadata from duplicates')
        header_data = normalized
        # If classifier_type is provided, filter columns by use_* flag
        use_flag = None
        if classifier_type:
            # Use exact casing as in schema: XGB, Cat, LGBM, Tree, Linear, NN, NB, SVM, KNN
            classifier_type = classifier_type.strip()
            # Map common lower/upper variants to canonical schema keys
            canonical_map = {
                'xgb': 'XGB',
                'cat': 'Cat',
                'lgbm': 'LGBM',
                'tree': 'Tree',
                'linear': 'Linear',
                'nn': 'NN',
                'nb': 'NB',
                'svm': 'SVM',
                'knn': 'KNN',
                'rf': 'RF',
                'randomforestclassifier': 'RF',
            }
            key = classifier_type.lower()
            canonical_type = canonical_map.get(key, classifier_type)
            use_flag = f"use_{canonical_type}"
        def is_true(val):
            return val is True or (isinstance(val, str) and val.lower() == 'true')
        # Determine if the schema actually uses this classifier-specific flag anywhere.
        schema_uses_flag = use_flag and any(use_flag in col for col in header_data)

        def is_used(col):
            if not use_flag:
                return is_true(col.get('X', False))
            # If the schema defines the classifier-specific use_flag anywhere, require it to be True.
            if schema_uses_flag:
                return is_true(col.get('X', False)) and is_true(col.get(use_flag, False))
            # Otherwise, fall back to including any column with X=True (legacy behavior)
            return is_true(col.get('X', False))

        # Debug: print classifier_type, use_flag, and value for each column
        logger.debug(f"[load_column_headers] classifier_type={classifier_type}, use_flag={use_flag}")
        for col in header_data:
            if use_flag and is_true(col.get('X', False)):
                logger.debug(f"  Col: {col['name']} | {use_flag}={col.get(use_flag, None)} | X={col.get('X', None)} | categorical={col.get('categorical', None)}")

        # Exclude raw *_Status columns from features if their OHE columns are present in the DataFrame
        raw_status_cols = set()
        ohe_status_cols = set()
        for col in header_data:
            if is_used(col) and col['name'].endswith('_Status'):
                raw_status_cols.add(sanitize_column_name(col['name']))
            # Collect OHE columns for status
            if 'ohe_from' in col and col['ohe_from'].endswith('_Status'):
                ohe_status_cols.add(sanitize_column_name(col['name']))

        # Only exclude raw *_Status columns if their OHE columns are present in the DataFrame
        feature_cols = []
        for col in header_data:
            orig_name = col['name']
            sanitized_name = sanitize_column_name(orig_name)
            if is_used(col):
                # Determine which column form (sanitized or original) exists in the provided DataFrame
                chosen_name = None
                # If sanitized form exists in df, prefer it
                if sanitized_name in df.columns:
                    chosen_name = sanitized_name
                elif orig_name in df.columns:
                    chosen_name = orig_name
                else:
                    # Default to original name so tests that expect original names still pass
                    chosen_name = orig_name
                if chosen_name in raw_status_cols:
                    # Exclude raw *_Status column ONLY if a corresponding OHE column actually exists in the DataFrame
                    has_ohe_in_df = any(((ohe_col in df.columns) or (sanitize_column_name(ohe_col) in df.columns)) for ohe_col in ohe_status_cols if ohe_col.startswith(sanitize_column_name(chosen_name)))
                    if has_ohe_in_df:
                        continue
                feature_cols.append(chosen_name)
        # Build categorical and target columns similar to feature_cols (respecting df column naming)
        categorical_cols = []
        target_cols = []
        for col in header_data:
            orig_name = col['name']
            sanitized_name = sanitize_column_name(orig_name)
            if is_true(col.get('categorical', False)) and (not use_flag or is_true(col.get(use_flag, False))):
                categorical_cols.append(sanitized_name if sanitized_name in df.columns else orig_name)
            if is_true(col.get('Y', False)):
                target_cols.append(sanitized_name if sanitized_name in df.columns else orig_name)
        # Collect derived OHE columns and the source columns that produce OHEs.
        # Include sanitized variations and both 'ohe' dict values and 'ohe_from' derived names to handle schema inconsistencies.
        ohe_cols = []
        ohe_source_cols = []
        for col in header_data:
            # Only include OHE columns if their use_flag is True (if classifier_type is set)
            if 'ohe' in col and isinstance(col['ohe'], dict):
                if not use_flag or col.get(use_flag, False) is True:
                    for raw_val, ohe_col in col['ohe'].items():
                        sanitized = sanitize_column_name(ohe_col)
                        if ohe_col not in ohe_cols:
                            ohe_cols.append(ohe_col)
                        if sanitized not in ohe_cols:
                            ohe_cols.append(sanitized)
                    if col['name'] not in ohe_source_cols:
                        ohe_source_cols.append(col['name'])
            if 'ohe_from' in col and 'ohe_key' in col:
                if not use_flag or col.get(use_flag, False) is True:
                    sanitized_name = sanitize_column_name(col['name'])
                    if col['name'] not in ohe_cols:
                        ohe_cols.append(col['name'])
                    if sanitized_name not in ohe_cols:
                        ohe_cols.append(sanitized_name)
                    if col['ohe_from'] not in ohe_source_cols:
                        ohe_source_cols.append(col['ohe_from'])

        logger.info(f"Schema loaded: {len(feature_cols)} features, {len(categorical_cols)} categorical, {len(target_cols)} targets.")

        # Validate that all defined feature columns exist in the DataFrame
        # Respect schema merge directives: if a feature is expected to be merged via
        # 'use_median_with' with a partner that exists in the DataFrame, allow its absence
        column_map = {col['name']: col for col in header_data}
        checkable_feature_cols = list(feature_cols)
        for col in list(feature_cols):
            if col not in df.columns:
                partner = column_map.get(col, {}).get('use_median_with')
                if partner:
                    print(f"[load_column_headers DEBUG] col={col!r}, partner={partner!r}, partner_in_df={partner in df.columns}")
                if partner and partner in df.columns:
                    # remove from strict check (presence of partner is sufficient)
                    checkable_feature_cols = [c for c in checkable_feature_cols if c != col]
                    # also proactively remove from reported feature list so downstream callers don't expect it
                    feature_cols = [c for c in feature_cols if c != col]

        # Inspect which of the 'checkable' columns are still missing after partner-based filtering
        missing_after_filter = set(checkable_feature_cols) - set(df.columns)
        print(f"[load_column_headers DEBUG] feature_cols={feature_cols}")
        print(f"[load_column_headers DEBUG] checkable_feature_cols={checkable_feature_cols}")
        print(f"[load_column_headers DEBUG] missing_after_filter={missing_after_filter}")
        # If any remaining missing columns are part of schema 'use_median_with' merge pairs, allow them to be absent
        allowed_by_merge = set()
        for m in list(missing_after_filter):
            mm = column_map.get(m, {})
            print(f"[load_column_headers DEBUG] checking missing m={m!r}, column_map_entry={mm}")
            if mm.get('use_median_with') or any((col.get('use_median_with') == m) for col in header_data):
                allowed_by_merge.add(m)
        if allowed_by_merge:
            print(f"[load_column_headers DEBUG] allowing missing merged columns: {allowed_by_merge}")
            checkable_feature_cols = [c for c in checkable_feature_cols if c not in allowed_by_merge]
            feature_cols = [c for c in feature_cols if c not in allowed_by_merge]
            missing_after_filter = missing_after_filter - allowed_by_merge

        try:
            check_df_columns(df, checkable_feature_cols)
        except ValueError as e:
            if allow_missing_columns:
                # The DataFrame may be synthetic or only include a subset of schema columns (tests use such dfs).
                # Log a warning and continue without strict validation to support testing and flexible inputs.
                logger.warning(f"DataFrame did not pass strict schema validation: {e}. Continuing without strict validation.")
                # Filter the columns to only those present in the DataFrame to avoid KeyErrors downstream
                feature_cols = [c for c in feature_cols if c in df.columns]
                categorical_cols = [c for c in categorical_cols if c in df.columns]
                target_cols = [c for c in target_cols if c in df.columns]
                # If no schema columns are present (e.g., synthetic DF), fall back to DataFrame columns
                if not feature_cols:
                    feature_cols = list(df.columns)
            else:
                # Re-raise for strict behavior
                logger.error(f"Error validating DataFrame columns against schema: {e}")
                raise

        # Filter OHE columns to those that actually exist in the DataFrame (normalize names)
        ohe_cols_present = [c for c in ohe_cols if c in df.columns]
        if len(ohe_cols_present) < len(ohe_cols):
            missing_ohe = set(ohe_cols) - set(ohe_cols_present)
            logger.debug(f"Some OHE columns from schema did not appear in the DataFrame and will be ignored: {missing_ohe}")

        return {
            "feature_cols": feature_cols, 
            "categorical_cols": categorical_cols, 
            "target_cols": target_cols,
            "ohe_cols": ohe_cols_present,
            "ohe_source_cols": ohe_source_cols
        }

    except FileNotFoundError as e:
        logger.error(f"Error: Column header .json file not found at '{column_headers_json}, {e}'.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from '{column_headers_json}'. Check for syntax errors.")
        raise
    except ValueError as e: # Raised by check_df_columns
        logger.error(f"Error validating DataFrame columns against schema: {e}")
        raise

def sanitize_column_name(col: str) -> str:
    """
    Replaces special characters in column names with underscores.
    Converts json data input, especially the offers, to more standardized camel case 
    names for each column.
    """
    mapping = {
       'Automatic Financing': 'AutomaticFinancing',
       '0% Unsecured Funding': '0UnsecuredFunding',
       'Debt Resolution': 'DebtResolution',
       'final_contract_tier':'final_contract_tier',
       'final_contract_amount':'final_contract_amount',
       'Automatic Financing_missing?':'AutomaticFinancing_missing?',
       'Automatic Financing_Score':'AutomaticFinancing_Score',
       'Automatic Financing_below_600?':'AutomaticFinancing_below_600_',
       'Automatic Financing_Status':'AutomaticFinancing_Status',
       'Automatic Financing_Amount':'AutomaticFinancing_Amount',
       'Automatic Financing_DebtToIncome':'AutomaticFinancing_DebtToIncome',
       'Automatic Financing_Details':'AutomaticFinancing_Details',
       'Automatic Financing_Contingencies':'AutomaticFinancing_Contingencies',
       '0% Unsecured Funding_missing?':'0UnsecuredFunding_missing_',
       '0% Unsecured Funding_Score':'0UnsecuredFunding_Score',
       '0% Unsecured Funding_below_600?':'0UnsecuredFunding_below_600_',
       '0% Unsecured Funding_Status':'0UnsecuredFunding_Status',
       '0% Unsecured Funding_Amount':'0UnsecuredFunding_Amount',
       '0% Unsecured Funding_DebtToIncome': '0UnsecuredFunding_DebtToIncome',
       '0% Unsecured Funding_Details':'0UnsecuredFunding_Details',
       '0% Unsecured Funding_Contingencies':'0UnsecuredFunding_Contingencies',
       'Debt Resolution_missing?':'DebtResolution_missing_',
       'Debt Resolution_Score': 'DebtResolution_Score',
       'Debt Resolution_below_600?':'DebtResolution_below_600_',
       'Debt Resolution_Status':'DebtResolution_Status',
       'Debt Resolution_Amount':'DebtResolution_Amount',
       'Debt Resolution_DebtToIncome':'DebtResolution_DebtToIncome',
       'Debt Resolution_Details':'DebtResolution_Details',
       'Debt Resolution_Contingencies':'DebtResolution_Contingencies',
       'Debt Resolution_score_missing?':'DebtResolution_score_missing_'
    }
    return re.sub(r'[^a-zA-Z0-9_]', '_', mapping.get(col, col))

def check_df_columns(df: pd.DataFrame, column_headers: List[str]) -> bool:
    """
    Check that a data frame is of high-enough quality to be used in ML model fitting.
    """
    bad_df = False
    
    # Check for missing columns first
    missing_columns = set(column_headers) - set(df.columns)
    if missing_columns:
        for col in missing_columns:
            logger.critical(f"DF is missing expected column: {col}")
        bad_df = True
        
    # Only check existing columns for NaN values
    existing_columns = set(column_headers) & set(df.columns)
    logger.debug(f"{df.to_string() = }")

    for c in existing_columns:
        try:
            column_data = df[c]
            # If duplicate column names exist in the DataFrame, df[c] may return a DataFrame
            if isinstance(column_data, pd.DataFrame):
                logger.warning(f"Duplicate column name detected in DataFrame for '{c}'; using first occurrence for validation.")
                # select the first occurrence (left-most)
                column_data = column_data.iloc[:, 0]

            # Now, whether we had duplicates or not, check for NaN values in the column
            try:
                nan_mask = column_data.isna()
            except Exception:
                # If column_data is not a Series for some reason, skip NaN checks but mark as bad
                logger.warning(f"Unable to check NaNs for column '{c}' (unexpected type: {type(column_data)})")
                bad_df = True
                continue

            if nan_mask.any():
                nan_indices = nan_mask[nan_mask].index
                logger.debug(f"NaN indices for column '{c}': {list(nan_indices)}")

                # Print the entire rows for these bad indices
                for i in nan_indices:
                    logger.debug(f"{i}: {df.loc[i]}")
                bad_df = True

        except KeyError as e:
            logger.critical(f"DF is missing expected column: {c}, error: {e}")
            bad_df = True


    if bad_df:
            raise ValueError("The DataFrame is not suitable for ML model fitting. Address the errors before proceeding.")
    return not bad_df

def map_color_to_cat(color: str) -> str:
    """
    Maps a color hex or color name to a canonical color category string.
    Used for PDF color extraction and feature engineering.
    Accepts hex codes (with or without #), common color names, or None.
    Returns one of: 'red', 'green', 'yellow', 'blue', 'gray', 'black', 'white', 'other', or 'unknown'.
    """
    if not color or not isinstance(color, str):
        return 'unknown'
    c = color.lower().strip()
    # Remove leading # if present
    if c.startswith('#'):
        c = c[1:]
    # Canonical color hexes (project-specific)
    color_map = {
        # Red
        'ff0000': 'red', 'e53935': 'red', 'd32f2f': 'red', 'c62828': 'red',
        # Green
        '00ff00': 'green', '43a047': 'green', '388e3c': 'green', '2e7d32': 'green',
        # Yellow
        'ffff00': 'yellow', 'fbc02d': 'yellow', 'f9a825': 'yellow', 'f57c00': 'yellow',
        # Blue
        '1976d2': 'blue', '1565c0': 'blue', '0d47a1': 'blue', '2196f3': 'blue',
        # Gray
        '9e9e9e': 'gray', 'bdbdbd': 'gray', '757575': 'gray', '616161': 'gray',
        # Black/White
        '000000': 'black', 'ffffff': 'white',
    }
    # Named color mapping
    name_map = {
        'red': 'red', 'green': 'green', 'yellow': 'yellow', 'blue': 'blue',
        'gray': 'gray', 'grey': 'gray', 'black': 'black', 'white': 'white',
        'orange': 'yellow', 'gold': 'yellow', 'navy': 'blue', 'lime': 'green',
    }
    if c in color_map:
        return color_map[c]
    if c in name_map:
        return name_map[c]
    # Try to match short hex (e.g., 'f00' for red)
    if len(c) == 3 and all(x in '0123456789abcdef' for x in c):
        if c == 'f00':
            return 'red'
        if c == '0f0':
            return 'green'
        if c == 'ff0':
            return 'yellow'
        if c == '00f':
            return 'blue'
        if c == '000':
            return 'black'
        if c == 'fff':
            return 'white'
    return 'other'


