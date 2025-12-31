"""
Converts a nested JSON file to a preprocessed CSV file, using a golden JSON schema
for order and standardization. The script dynamically applies a variety of
transformations based on the provided schema definition.
"""

import json
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.utils import check_df_columns, sanitize_column_name
import re
import numpy as np
try:
    from sklearn.preprocessing import LabelEncoder
except Exception:  # pragma: no cover - sklearn may not be installed in test env
    LabelEncoder = None

logger = logging.getLogger(__name__)

# --- Helpers ---
def clean_and_normalize_numeric(value: Any) -> Optional[float]:
    """Cleans and converts a string or numeric value to a float. Returns None if not parseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "none", "nan"}:
        return None
    s = re.sub(r'[$,\s]', '', s)
    try:
        return float(s)
    except Exception:
        return None

def generate_auto_labels(series: pd.Series, generation_methods: List[str], min_freq: float = 0.02) -> pd.Series:
    """Generate auto labels based on grouped and frequency methods."""
    logger.info(f"Generating auto labels for {series.name} with methods {generation_methods}")
    result = series.astype(str).copy()
    for m in generation_methods:
        if m == "grouped":
            min_count = max(5, int(min_freq * len(series)))
            counts = result.value_counts()
            rare = counts[counts < min_count].index
            result = result.apply(lambda x: "other" if x in rare else x)
            logger.debug(f"Grouped rare classes (<{min_count} occurrences) into 'other': {rare.tolist()}")
        if m == "frequency":
            counts = result.value_counts()
            freq_map = {val: i for i, (val, _) in enumerate(counts.items())}
            result = result.map(freq_map)
            logger.debug(f"Frequency-based mapping: {freq_map}")
    return result.fillna(-1).astype(int)

# --- Contract processing ---
def _process_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a single contract dictionary."""
    tier_mapping = {
        '0.045': 'A - Tier 1 VIP', '0.095': 'A - Tier 1', '0.24': 'A - Tier 2',
        '0.34': 'A - Tier 3', '0.44': 'A - Tier 4', '0.165': 'A - Tier 1 Low APR',
        '0.31': 'A - Tier 2 Low APR', '0.41': 'A - Tier 3 Low APR',
        '0.51': 'A - Tier 4 Low APR', '0.12': 'B - Tier 1 Prime'
    }

    if not contract:
        return {}

    tier_key = str(contract.get('tier')) if contract.get('tier') is not None else None
    tier_name = tier_mapping.get(tier_key) if tier_key is not None else None
    return {
        'status': contract.get('status'),
        'tier': contract.get('tier'),
        'amount': contract.get('amount'),
        'tier_name': tier_name
    }


# --- Offers processing ---

def _process_offers(offers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Processes the list of offers, handling missing offers, data anomalies, and feature engineering."""
    offer_map: Dict[str, Dict[str, Any]] = {
        sanitize_column_name(str(o["Name"])): o
        for o in offers
        if isinstance(o, dict) and o.get("Name") is not None
    }
    new_features: Dict[str, Any] = {}
#20250826    logger.debug(f"0ufs01: /ok, strings/{offer_map = }")

    offer_types = {
        'Automatic Financing': {'Details': 'NA', 'Contingencies': 'NA', 'Name':'Automatic Financing'},
        '0% Unsecured Funding': {'Details': 'NA', 'Contingencies': '-1', 'Name':'0% Unsecured Funding'},
        'Debt Resolution': {'Details': '', 'Contingencies': '', 'Name':'Debt Resolution'}
    }
    new_features = {}

    for name, defaults in offer_types.items():
        sanitized_name = sanitize_column_name(name)
        offer = offer_map.get(sanitized_name)
        logger.debug(f"NaNs00: //{offer = } for {sanitized_name}")
        is_missing = offer is None
#20250826        logger.debug(f"0ufs02a: /ok, strings/{offer = }")
#20250826        logger.debug(f"0ufs02b: /ok, strings/{offer_map.get(sanitized_name) = }")
#20250826        logger.debug(f"0ufs02c: /ok, strings/{offer_map.get('0UnsecuredFunding') = }")


        new_features[f"{sanitized_name}_missing_"] = 1 if is_missing else 0
        new_features[f"{sanitized_name}_Name"] = name if not is_missing else 'NA'

        if is_missing:
            # Represent missing offer numerics as NaN, keep missing flag and Name/Status indicators
            new_features[f"{sanitized_name}_Score"] = np.nan
            new_features[f"{sanitized_name}_below_600_"] = np.nan
            new_features[f"{sanitized_name}_Status"] = 'NA'
            new_features[f"{sanitized_name}_Amount"] = np.nan
            new_features[f"{sanitized_name}_DebtToIncome"] = np.nan
            new_features[f"{sanitized_name}_Details"] = defaults['Details']
            new_features[f"{sanitized_name}_Contingencies"] = defaults['Contingencies']
            if sanitized_name == 'DebtResolution':
                new_features[f"{sanitized_name}_score_missing?"] = 1
            new_features[f"{sanitized_name}_PayD"] = np.nan
            new_features[f"{sanitized_name}_Collections"] = np.nan
            new_features[f"{sanitized_name}_score_missing_"] = np.nan
            continue
#20250826        logger.debug(f"0ufs03: /ok, strings/{offer = }")

        # Process non-missing offer
        status = str(offer.get('Status') or 'NA')
        if sanitized_name in {'0UnsecuredFunding', 'DebtResolution'} and status.strip() in {"", "NA"}:
            status = 'Declined'
#20250826        logger.debug(f"0ufs04: /ok, strings/{offer = }")

        amount = clean_and_normalize_numeric(offer.get('Amount')) or 0.0
        score = offer.get('Score')

        below_600_flag: int = 0
        score_missing_flag: int = 0
        numeric_score: Optional[float] = None

        if isinstance(score, str) and score.strip().lower() == "below 600":
            numeric_score = 599.0
            below_600_flag = 1
        elif isinstance(score, (int, float)) and score < 600:
            numeric_score = float(score)
            below_600_flag = 1
        else:
            numeric_score = clean_and_normalize_numeric(score)

        score_missing_flag = 0
        if sanitized_name == "DebtResolution" and status == "Declined" and numeric_score is None:
            numeric_score = -999.0
            score_missing_flag = 1

        dti = clean_and_normalize_numeric(offer.get("DebtToIncome"))
        if sanitized_name == "DebtResolution" and status == "Declined" and dti is None:
            dti = -999.0

        if sanitized_name == "0UnsecuredFunding":
            contingencies = str(offer.get("Contingencies") or "")
            payd_match = re.search(r'PayD-\$([\d,]+)', contingencies)
            coll_match = re.search(r'Collections-\$([\d,]+)', contingencies)
            payd_val = clean_and_normalize_numeric(payd_match.group(1)) if payd_match else 0.0
            logger.debug(f"PayD00: /numeric/ {contingencies = }, {payd_match = }, {payd_val = }")
            coll_val = clean_and_normalize_numeric(coll_match.group(1)) if coll_match else 0.01
            new_features["0UnsecuredFunding_PayD"] = payd_val
            new_features["0UnsecuredFunding_Collections"] = coll_val

        new_features[f"{sanitized_name}_Score"] = numeric_score if numeric_score is not None else -1
        new_features[f"{sanitized_name}_below_600_"] = below_600_flag
        if sanitized_name == "DebtResolution":
            new_features[f"{sanitized_name}_score_missing_"] = score_missing_flag

        new_features[f"{sanitized_name}_Name"] = offer.get("Name", defaults["Name"])
        new_features[f"{sanitized_name}_Status"] = status
        new_features[f"{sanitized_name}_Amount"] = amount
        new_features[f"{sanitized_name}_DebtToIncome"] = dti if dti is not None else -1
        new_features[f"{sanitized_name}_Details"] = offer.get("Details", defaults["Details"])
        new_features[f"{sanitized_name}_Contingencies"] = offer.get("Contingencies", defaults["Contingencies"])
        logger.debug(f"NaNs01: //{new_features = }")

    return new_features


# --- Schema utils ---

def _load_golden_schema(golden_json_path: Path) -> List[Dict[str, Any]]:
    """Load the golden schema from JSON file."""
    with golden_json_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _parse_schema(schema: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Parse schema into sorted schema and column map."""
    #20250825column_map = {col['name']: col for col in schema if col.get('x_value')=="True"}
    column_map = {col['name']: col for col in schema}
    sorted_schema: List[Dict[str, Any]] = []
    derived_columns = {c['name'] for c in schema if any(k in c for k in ['labels_from', 'mapped_from', 'ohe_from'])}
    source_columns = [c for c in schema if c['name'] not in derived_columns]
    sorted_schema.extend(source_columns)
    sorted_schema.extend([c for c in schema if c['name'] in derived_columns])
    return sorted_schema, column_map


def _apply_labeling_and_mapping(df: pd.DataFrame, schema_item: Dict[str, Any], column_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    col_name = schema_item['name']
    source_col_name = schema_item.get('labels_from', '') or schema_item.get('mapped_from', '')
    
    logger.info(f"Processing {col_name} from {source_col_name}")
    logger.debug(f"label00 - **{schema_item['name']}**{source_col_name}**{col_name}**{column_map.get(source_col_name, {}).get('labels', {})}**\n{schema_item = }") #type: ignore

    if source_col_name not in df.columns:
        logger.critical(f"Error parsing column_headers json file.'{schema_item.get('name')}' referenced '{source_col_name}'.")
        raise ValueError(f"Source column {source_col_name} not found in DataFrame but it is specified as the source of {col_name}. Aborting")

    # Store original data for debugging
    original_data = df[source_col_name].copy() if source_col_name else pd.Series([None]*len(df))
    
    if schema_item.get('labels_from'):
        source_schema = column_map.get(source_col_name, {})
        if source_schema.get('labels') == 'auto-generated' and source_schema.get('generation_methods'):
            df[col_name] = generate_auto_labels(
                df[source_col_name],
                generation_methods=source_schema.get('generation_methods', []),
                min_freq=0.02
            )
            logger.info(f"Generated auto-labels for {col_name} using {source_schema.get('generation_methods')}")
        else:
            mapping_dict = source_schema.get('labels', {})
            logger.debug(f"label01b - **{mapping_dict = }")
            fill_val = -1 if 'labels' in source_schema or 'labels_from' in schema_item else 'NA'
            df[col_name] = pd.to_numeric(
                df[source_col_name].astype(str).map(mapping_dict),
                errors='coerce'
            ).fillna(fill_val).astype(int, errors='ignore')
            
    elif schema_item.get('mapped_from'):
        mapping_dict = column_map.get(source_col_name, {}).get('mapping', {})
        fill_val = 'NA'
        df[col_name] = df[source_col_name].astype(str).map(mapping_dict).fillna(fill_val)

    else:
        mapping_dict = schema_item.get('labels') or schema_item.get('mapping', {})
        fill_val = -1 if 'labels' in schema_item else 'NA'
        df[col_name] = pd.to_numeric(
            df[source_col_name].astype(str).map(mapping_dict),
            errors='coerce'
        ).fillna(fill_val).astype(int, errors='ignore') if 'labels' in schema_item else \
            df[source_col_name].astype(str).map(mapping_dict).fillna(fill_val)

    # Efficient debugging - only for failed mappings
    if logger.isEnabledFor(logging.DEBUG) and source_col_name:
        failed_mappings = df[col_name].isin([-1, 'NA'])
        if failed_mappings.any():
            failed_records = df.loc[failed_mappings, ['record_id', col_name]].copy()
            failed_records['original'] = original_data.loc[failed_mappings]
            logger.debug(f"Failed mappings for {col_name}:")
            for _, row in failed_records.iterrows():  # Only show first 5
                logger.debug(f"  {row['original']} -> {row[col_name]} (record_id={row['record_id']})")
    return df

def _apply_ohe(df: pd.DataFrame, schema_item: Dict[str, Any], column_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Apply one-hot encoding based on schema."""
    col_name = schema_item['name']
    if 'ohe' in schema_item:
        for raw_val, ohe_col in schema_item['ohe'].items():
            if ohe_col not in df.columns:
                df[ohe_col] = 0
            try:
#20250826                logger.debug(f"Applying OHE for {col_name}: {raw_val} -> {ohe_col}")
                df[ohe_col] = (df[col_name].astype(str) == str(raw_val)).astype(int)
            except KeyError as e:
                raise KeyError(f"Error applying OHE for {col_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error applying OHE for {col_name}: {e}")
#20250826             logger.debug(f"here7a - /_flag?/ - **{ohe_col}**{col_name}**{raw_val}**")
#20250826        for c in df.columns:
#20250826             logger.debug(f"here7b - /_flag?/ - **{c}**")

    elif 'ohe_from' in schema_item and 'ohe_key' in schema_item:
        src = schema_item['ohe_from']
        key = schema_item['ohe_key']
        if src in df.columns:
            df[col_name] = (df[src].astype(str) == str(key)).astype(int)
#20250826             logger.debug(f"here8 - /_flag?/ - **{col_name}**{src}**{key}**")
    return df


def _get_final_columns(schema: List[Dict[str, Any]]) -> List[str]:
    """Get final column names from schema."""
    return [col['name'] for col in schema]


# --- Main ---
def flatten_weaviate_data(file_path: Path, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten Weaviate JSON data into a DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise ValueError(f"Could not read or parse JSON file {file_path}: {e}")
    records = full_data.get("data", [])
#20250826     logger.debug(f"here1: (raw, so spaces) {records = }")
    flattened_records: List[Dict[str, Any]] = []
    for record in records:
        flat: Dict[str, Any] = {
            "record_id": record.get("record_id"),
            "user_initials": record.get("user_initials"),
            "DebtToIncome": clean_and_normalize_numeric(record.get("prefi_data", {}).get("DataEnhance", {}).get("DebtToIncome")),
            "final_contract_status": None,
            "final_contract_tier": None,
            "final_contract_amount": None,
        }

        # Process contracts if they exist
        if 'contracts' in record and record['contracts']:
            most_recent = sorted(record['contracts'], key=lambda x: pd.to_datetime(x.get('created_at')), reverse=True)[0]

            # Handle contract status
            status = most_recent.get('status')
            flat['final_contract_status'] = str(status) if status is not None else 'NA'

            # Handle contract tier - ensure it's a number or -1
            tier = clean_and_normalize_numeric(most_recent.get('tier'))
            flat['final_contract_tier'] = tier if tier is not None else -1

            # Handle contract amount - ensure it's a number or -1
            amount = clean_and_normalize_numeric(most_recent.get('amount'))
            flat['final_contract_amount'] = amount if amount is not None else -1

            # Capture contract created_at timestamp (keep original string for now)
            flat['final_contract_created_at'] = most_recent.get('created_at')

        # Process offers
        offers = record.get('prefi_data', {}).get('Offers', [])
        flat.update(_process_offers(offers))
        flattened_records.append(flat)

    # Create DataFrame and ensure no NaN values
    df = pd.DataFrame(flattened_records)

    # Fill any remaining NaN values before validation
    #df = df.fillna(-1)  # or appropriate default values

    # Validate columns match schema
    ## Remove the columns that haven't been added yet
    ### One-hot encoding
    ohe_cols = {item.get('name') for item in schema if 'ohe_from' in item.keys()}

    ### Labels
    label_cols = {item.get('name') for item in schema if 'labels_from' in item.keys()}
    
    ### Mapping
    mapping_cols = {item.get('name') for item in schema if 'mapped_from' in item.keys()}

    test_columns = set(_get_final_columns(schema))-ohe_cols-label_cols-mapping_cols
    if not check_df_columns(df, list(test_columns)):
        logger.critical("DataFrame columns do not match schema.")
        raise ValueError(f"Values in {file_path} JSON records do not match schema.")

    return df

def preprocess_dataframe(df: pd.DataFrame, schema: List[Dict[str, Any]], column_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Preprocess DataFrame according to schema."""
    processed = df.copy()
    for item in schema:
        #20250826bif any(k in item for k in ["labels", "mapping", "labels_from", "mapped_from"]):
        if any(k in item for k in ["labels_from", "mapped_from"]):
            processed = _apply_labeling_and_mapping(processed, item, column_map)
        if "ohe" in item or "ohe_from" in item:
#20250826             logger.debug(f"here6 - /*_flag_/ - {item = }")
            processed = _apply_ohe(processed, item, column_map)
    final_cols = _get_final_columns(schema)
#20250826     logger.debug(f"here10 - *_flag_ - {final_cols = }")
#20250826     logger.debug(f"here11 - *_flag? - {processed.columns = }")
    # Reindex to the final schema and fill numeric-style missing values with 0
    processed = processed.reindex(columns=final_cols, fill_value=0)

    # Convert any datetime-like columns (e.g., *_created_at) to pandas datetime dtype
    for col in processed.columns:
        if col.endswith('_created_at') and col in processed.columns:
            try:
                processed[col] = pd.to_datetime(processed[col], errors='coerce')
            except Exception:
                # leave as-is if conversion fails
                pass

    # Ensure string-like derived columns (names, details, contingencies, status)
    # do not contain NaNs or numeric sentinels. Fill with 'NA' and coerce to object.
    string_like_keys = ('Name', 'Details', 'Contingencies', 'Status')

    # Only apply string-like cleanup to 'source' columns (e.g., AutomaticFinancing_Status)
    for col in final_cols:
        if any(col.endswith(f"_{k}") or col == k for k in string_like_keys) and col in processed.columns:
            if processed[col].isnull().any() or processed[col].dtype.kind in ('f', 'i'):
                # Replace common numeric sentinel values with 'NA' as well
                processed[col] = processed[col].replace({-1: 'NA', -999.0: 'NA', 0: 'NA'})
                processed[col] = processed[col].fillna('NA').astype(object)

    # Enforce OHE dtypes: convert OHE generated columns to integers (0/1) and replace 'NA' string sentinels with 0
    ohe_cols = []
    for item in schema:
        if 'ohe' in item and isinstance(item['ohe'], dict):
            for _, derived_col in item['ohe'].items():
                ohe_cols.append(derived_col)
        if 'ohe_from' in item and 'ohe_key' in item:
            ohe_cols.append(item['name'])

    for col in set(ohe_cols):
        if col in processed.columns:
            try:
                # Replace string sentinels with 0 and cast to int
                processed[col] = processed[col].replace({'NA': 0})
                processed[col] = processed[col].fillna(0)
                # Attempt to coerce numeric values to ints
                processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0).astype(int)
            except Exception:
                # As a fallback, ensure 0/1 by comparing to strings
                processed[col] = (processed[col].astype(str) == 'True').astype(int)

    # Coerce numeric-like feature columns to numeric types where appropriate
    for item in schema:
        name = item.get('name')
        if item.get('X') == 'True' and item.get('categorical') == 'False' and name in processed.columns:
            # If column appears object-like (strings with currency or commas), try to normalize
            if processed[name].dtype == object:
                try:
                    processed[name] = processed[name].apply(lambda v: clean_and_normalize_numeric(v) if v is not None else None)
                    processed[name] = pd.to_numeric(processed[name], errors='coerce').fillna(-1)
                except Exception:
                    # If coercion fails, leave as-is
                    pass

    # Final sweep to ensure no NaN/inf values remain according to schema rules
    processed = ensure_no_nans(processed, schema)

    return processed


def ensure_no_nans(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    """Ensure no NaN/inf values remain in a preprocessed DataFrame.

    - Numeric columns: replace NaN/inf with -1
    - Object/string columns: replace NaN with 'NA'
    - OHE columns: replace NaN with 0 and coerce to int
    Returns the cleaned DataFrame and logs replacements.
    """
    cleaned = df.copy()
    # Identify OHE columns from schema
    ohe_cols = set()
    for item in schema:
        if 'ohe' in item and isinstance(item['ohe'], dict):
            for _, derived_col in item['ohe'].items():
                ohe_cols.add(derived_col)
        if 'ohe_from' in item and 'ohe_key' in item:
            ohe_cols.add(item['name'])

    # Replace numeric NaNs/infs
    num_cols = cleaned.select_dtypes(include=['number']).columns
    for c in num_cols:
        before_na = int(cleaned[c].isnull().sum() + np.isinf(cleaned[c]).sum())
        if before_na:
            cleaned[c] = pd.to_numeric(cleaned[c], errors='coerce')
            cleaned[c] = cleaned[c].replace([np.inf, -np.inf], np.nan).fillna(-1)
            logger.info(f"Replaced {before_na} NaN/inf(s) in numeric column '{c}' with -1")

    # Replace object NaNs
    obj_cols = cleaned.select_dtypes(include=['object']).columns
    for c in obj_cols:
        before_na = int(cleaned[c].isnull().sum())
        if before_na:
            cleaned[c] = cleaned[c].fillna('NA')
            logger.info(f"Replaced {before_na} NaN(s) in object column '{c}' with 'NA'")

    # Enforce OHE columns are 0/1 ints
    for c in ohe_cols:
        if c in cleaned.columns:
            before_na = int(cleaned[c].isnull().sum())
            if before_na:
                cleaned[c] = cleaned[c].fillna(0)
                logger.info(f"Replaced {before_na} NaN(s) in OHE column '{c}' with 0")
            # Replace common sentinels with 0 (e.g., -1, 'NA') then coerce to 0/1 ints
            cleaned[c] = cleaned[c].replace({-1: 0, '-1': 0, 'NA': 0})
            try:
                cleaned[c] = pd.to_numeric(cleaned[c], errors='coerce').fillna(0).astype(int)
                # Clamp values to 0/1
                cleaned[c] = cleaned[c].apply(lambda v: 1 if v else 0).astype(int)
            except Exception:
                cleaned[c] = (cleaned[c].astype(str) == 'True').astype(int)

    return cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a nested JSON file to preprocessed CSV.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--column_headers_json", required=True, type=Path)
    parser.add_argument("-o", "--output_file", required=True, type=Path)
    args = parser.parse_args()

    schema = _load_golden_schema(args.column_headers_json)
    sorted_schema, column_map = _parse_schema(schema)

    df = flatten_weaviate_data(args.input_file, sorted_schema)
#20250826    logger.debug(f"0ufs00: {df['0UnsecuredFunding_Status'] = }")
    logger.debug(f"PayD01: {df['0UnsecuredFunding_Status'] = }")

    final = preprocess_dataframe(df, sorted_schema, column_map)
#20250826    for c in final.columns:
#20250826         logger.debug(f"here12 - /CamelCase/ - **{c}**")
    check_df_columns(final, _get_final_columns(sorted_schema))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_file, index=False)
    logger.info(f"Saved {final.shape[0]} rows, {final.shape[1]} cols to {args.output_file}")

    sys.exit(0)