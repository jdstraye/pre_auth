"""
Compare columns of one or more CSV files against a golden JSON file and print a structured report.

This script checks if the columns in the CSV files match the columns specified in the golden JSON file.
It distinguishes between required and optional columns based on the 'X' and 'Y' fields in the JSON file.
The output is a structured report displaying each column from the golden JSON and its status in each CSV file.

Usage:
    python compare_columns.py -csv <file1.csv> -csv <file2.csv> -gold_json <golden.json>

Arguments:
    -csv: Path to a CSV file. Can be specified multiple times for multiple files.
    -gold_json: Path to the golden JSON file.

Golden JSON format:
    A list of dictionaries, each representing a column. Each dictionary should have the following keys:
        - name: The name of the column.
        - X: A string indicating whether the column is required ('True') or not ('False').
        - Y: A string indicating whether the column is required ('True') or not ('False').

Example golden JSON:
    [
        {"name": "column1", "X": "True", "Y": "False"},
        {"name": "column2", "X": "False", "Y": "True"}
    ]
"""
#
# compare_columns_v4.py
#
# Purpose: Compares the column headers of one or more CSV files against a golden JSON file,
#          distinguishing between required and optional columns. It generates a detailed
#          report with a specific side-by-side comparison format.
#
# Author: SW Engineer, IP Developer
# Date: 2025-08-24
#
import pandas as pd
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _load_data(csv_paths: List[Path], gold_json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Loads golden column headers from a JSON file and columns from one or more CSVs.
    
    Args:
        csv_paths (List[Path]): A list of pathlib.Path objects for the CSV files.
        gold_json_path (Path): A pathlib.Path object for the golden JSON file.
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, List[str]]]: 
            A tuple containing:
            1. A list of dictionaries from the golden JSON.
            2. A dictionary mapping each CSV filename to its list of column names.
            
    Raises:
        SystemExit: If a critical error (e.g., FileNotFoundError, JSONDecodeError) occurs.
    """
    logger.info("Starting column comparison...")
    
    try:
        with gold_json_path.open('r', encoding='utf-8') as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Golden JSON file '{gold_json_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode '{gold_json_path}'. Check for syntax errors.")
        sys.exit(1)
    
    csv_columns: Dict[str, List[str]] = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, dtype=str)
            csv_columns[csv_path.name] = df.columns.tolist()
        except FileNotFoundError:
            logger.error(f"Error: CSV file '{csv_path}' not found. Skipping file.")
        except pd.errors.EmptyDataError:
            logger.warning(f"Warning: CSV file '{csv_path}' is empty. Skipping file.")
        except Exception as e:
            logger.error(f"Error reading CSV file '{csv_path}': {e}. Skipping file.")
    
    if not csv_columns:
        logger.error("No valid CSV files were loaded. Exiting.")
        sys.exit(1)

    return golden_data, csv_columns

def _analyze_columns(golden_data: List[Dict[str, Any]], csv_columns: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Analyzes and compares CSV columns against the golden standard.
    
    Returns a structured dictionary detailing the status of each CSV file's columns.
    
    Args:
        golden_data (List[Dict[str, Any]]): A list of dictionaries from the golden JSON.
        csv_columns (Dict[str, List[str]]): A dictionary of CSV filenames to their columns.
        
    Returns:
        Dict[str, Any]: A dictionary containing the analysis results.
    """
    golden_column_names: Set[str] = {col['name'] for col in golden_data}
    required_columns: Set[str] = {col['name'] for col in golden_data if col.get('X') == 'True' or col.get('Y') == 'True'}
    
    analysis_results: Dict[str, Any] = {
        'all_required_present': True,
        'csv_reports': {}
    }
    
    for filename, cols in csv_columns.items():
        csv_cols_set: Set[str] = set(cols)
        
        missing_cols: Set[str] = golden_column_names - csv_cols_set
        extra_cols: Set[str] = csv_cols_set - golden_column_names
        
        missing_required: Set[str] = missing_cols.intersection(required_columns)
        missing_optional: Set[str] = missing_cols - required_columns
        
        status = 'PASS' if not missing_required and not extra_cols else 'FAIL'
        
        if missing_required:
            analysis_results['all_required_present'] = False

        analysis_results['csv_reports'][filename] = {
            'status': status,
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'extra': extra_cols,
            'csv_cols_present': csv_cols_set, # Store the set of present columns for the report
        }
        
    return analysis_results

def generate_and_print_report(golden_data: List[Dict[str, Any]], analysis_results: Dict[str, Any], gold_json_path: Path):
    """
    Generates and prints a user-friendly report based on the analysis results.
    
    Args:
        golden_data (List[Dict[str, Any]]): The golden column data.
        analysis_results (Dict[str, Any]): The structured analysis data.
        gold_json_path (Path): Path to the golden JSON file for display purposes.
    """
    logger.info(f"Comparison report against golden standard: '{gold_json_path.name}'")
    
    all_passed = True
    for filename, report in analysis_results['csv_reports'].items():
        
        if report['status'] == 'PASS':
            logger.info(f"✅ PASSED: All columns in '{filename}' match the golden standard.")
        else:
            all_passed = False
            logger.error(f"❌ FAILED: Discrepancies found in '{filename}'. See detailed report below.")
            
            # --- Construct the new table format ---
            discrepancy_table = []
            
            # Get the set of required and optional golden columns
            golden_required = {col['name'] for col in golden_data if col.get('X') == 'True' or col.get('Y') == 'True'}
            golden_optional = {col['name'] for col in golden_data} - golden_required

            # 1. Add required columns
            for col_name in sorted(list(golden_required)):
                csv_presence = col_name if col_name in report['csv_cols_present'] else ''
                discrepancy_table.append(['REQ', col_name, csv_presence])

            # 2. Add optional columns that are present in the CSV
            for col_name in sorted(list(golden_optional.intersection(report['csv_cols_present']))):
                discrepancy_table.append(['OPT', col_name, col_name])
            
            # 3. Add a separator for extra columns
            if report['extra']:
                discrepancy_table.append([]) # Blank row for spacing
            
            # 4. Add extra columns from the CSV
            for col_name in sorted(list(report['extra'])):
                discrepancy_table.append(['', '', col_name])

            # Print the formatted table
            if discrepancy_table:
                header = ['TYPE', 'GOLDEN COLUMN', filename]
                print_formatted_table([header] + discrepancy_table)
    
    # Final summary message and exit code
    if all_passed:
        logger.info("🎉 All specified CSV files passed the column comparison.")
        sys.exit(0)
    else:
        logger.error("🚫 One or more CSV files failed the column comparison.")
        if not analysis_results['all_required_present']:
             logger.error("Required columns were missing in at least one file.")
             sys.exit(1)
        else:
            sys.exit(0)

def print_formatted_table(data: List[List[str]]) -> None:
    """Prints a list of lists as a formatted, aligned table."""
    if not data:
        return

    # Filter out empty rows before calculating widths
    non_empty_data = [row for row in data if row]
    if not non_empty_data:
        return

    column_widths = [max(len(str(item)) for item in col) for col in zip(*non_empty_data)]
    
    # Print the header (first row)
    header = data[0]
    print(" | ".join(f"{item:<{width}}" for item, width in zip(header, column_widths)))
    print("-" * (sum(column_widths) + 2 * (len(column_widths) - 1)))
    
    for row in data[1:]: # Start from the second row
        if not row:
            print()
        else:
            print(" | ".join(f"{item:<{width}}" for item, width in zip(row, column_widths)))
    print()

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CSV columns against a golden JSON file.')
    parser.add_argument('-csv', action='append', help='Path to a CSV file. Can be specified multiple times.')
    parser.add_argument('-gold_json', required=True, help='Path to the golden JSON file.')
    args = parser.parse_args()

    if not args.csv:
        logger.error("Error: At least one CSV file path must be provided.")
        sys.exit(1)

    csv_paths = [Path(csv_path) for csv_path in args.csv]
    gold_json_path = Path(args.gold_json)

    golden_data, csv_columns = _load_data(csv_paths, gold_json_path)
    analysis_results = _analyze_columns(golden_data, csv_columns)

    generate_and_print_report(golden_data, analysis_results, gold_json_path)
    
    sys.exit(0)  # Exit with success code if reached here