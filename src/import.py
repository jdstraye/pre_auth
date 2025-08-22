import json
import sys
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single contract dictionary and return a re-formatted contract with tier_name.

    Args:
        contract (Dict[str, Any]): The contract data to process.
    Returns:
        the initial contract with an additional key, 'tier_name'
    """
    def retn_contract(contract:Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
        return {
            'status': contract.get('status'),
            'tier': contract.get('tier'),
            'amount': contract.get('amount'),
            'tier_name': name
        }

    tier_mapping = {
        '0.045': 'A - Tier 1 VIP',
        '0.095': 'A - Tier 1', 
        '0.24':  'A - Tier 2',
        '0.34':  'A - Tier 3',
        '0.44':  'A - Tier 4',
        '0.165': 'A - Tier 1 Low APR',
        '0.31':  'A - Tier 2 Low APR',
        '0.41':  'A - Tier 3 Low APR',
        '0.51':  'A - Tier 4 Low APR',
        '0.12':  'B - Tier 1 Prime'
    }

    # Check if the contract is empty or None
    if not contract:
        raise ValueError("No contract data available.")
    if contract.get('tier') is None:
        # Default or handle missing tier
        logger.warning(f"Missing tier information in contract, id:{contract.get('contract_id')}.")
        contract['tier_name'] = None
        return retn_contract(contract, "null")

    tier_key = str(contract.get('tier')) if contract.get('tier') is not None else None
    tier_name = tier_mapping.get(tier_key) if tier_key is not None else None

    return retn_contract(contract, tier_name)

def flatten_weaviate_data(file_path: Path) -> pd.DataFrame:
    """
    Reads a JSON file with the new prefi_weaviate schema, flattens its nested structure,
    and converts it into a pandas DataFrame.

    This function extracts data from the top-level 'data' list, the 'prefi_data' object,
    the nested 'Offers' list, and the nested 'contracts' list to create a
    comprehensive tabular representation for a machine learning classifier.

    Args:
        file_path (Path): The path to the input JSON file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the flattened data.
    """
    try:
        # Step 1: Read and parse the JSON file with robust error handling
        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # Access the 'data' list, which contains the records we want to flatten
        records = full_data.get("data", [])
        if not records:
            logger.error("JSON file does not contain a 'data' key or it is empty.")
            raise ValueError("JSON file does not contain a 'data' key or it is empty.")

        # Step 2: Manually flatten the nested lists and objects for each record
        flattened_records: List[Dict[str, Any]] = []

        for record in records:
            # Extract top-level and directly-nested fields
            record_id = record.get("record_id")
            user_initials = record.get("user_initials")

            # Extract nested data from 'prefi_data'
            prefi_data = record.get("prefi_data", {})
            debt_to_income = prefi_data.get("DataEnhance", {}).get("DebtToIncome")

            # Extract data from the 'Offers' list
            offers = prefi_data.get("Offers", [])

            # Process all offers for this record
            offer_details = offers

            # Extract data from the 'Contracts' list
            contracts = record.get("contracts", [])

            # Process all contracts for this record
            contract_details = {}

            # Take only the final contract in the list
            for contract in contracts:
                contract_details = process_contract(contract)
                

            # Create a flattened record with all the details
            flattened_record: Dict[str, Any] = {
                "record_id": record_id,
                "user_initials": user_initials,
                "DebtToIncome": debt_to_income,
                "final_contract_status": contract_details.get("status") if contract_details else None,
                "final_contract_tier": contract_details.get("tier") if contract_details else None,
                "final_contract_tier_name": contract_details.get('tier_name')  if contract_details else None
            }
            # Add offer details as separate columns, including the offer name in the key
            for i, offer in enumerate(offer_details):
                offer_name = offer.get('Name', f'offer_{i+1}')
                for key, value in offer.items():
                   flattened_record[f"{offer_name}_{key}"] = value

            flattened_records.append(flattened_record)

        return pd.DataFrame(flattened_records)

    except FileNotFoundError as e:
        logger.error(f"Error: The file '{file_path}' was not found.")
        raise FileNotFoundError(f'{e}')
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode the JSON file '{file_path}'. Check for syntax errors.")
        raise # Re-raise the original exception
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise RuntimeError (f'{e}')

    return pd.DataFrame()

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Convert a nested JSON file to a flattened CSV table.')

    # Add command-line arguments
    parser.add_argument('input_file', type=Path, help='Path to the input JSON file.')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Path to the output CSV file. Defaults to a filename in the same directory as the input file.')

    # Parse the arguments
    args = parser.parse_args()

    # Get the file paths from the parsed arguments
    input_path = args.input_file

    # If no output path is specified, create a default one
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.parent / f"{input_path.stem}_flattened.csv"

    # Call the core function with the provided input path
    table_df = flatten_weaviate_data(input_path)

    if not table_df.empty:
        logger.info(f"Successfully converted JSON from '{input_path}' to a pandas DataFrame.")
        logger.info(f"The first 5 rows of the DataFrame:\n{table_df.head()}")

        try:
            # Save the DataFrame to a CSV file
            table_df.to_csv(output_path, index=False)
            logger.info(f"\nSuccessfully saved DataFrame to '{output_path}'")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error: Could not save the output file. Reason: {e}")
            sys.exit(1)
    else:
        logger.critical(f"Error: Nothing saved because table_df is empty")
        raise RuntimeError("table_df is empty")
