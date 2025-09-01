"""
Helper functions common throughout the PreAuth ML model-making project.
"""
""" 
Creates a custom log handler that creates a fresh log file with a date in the name but also creates a symlink to the latest log file.
"""
import logging
import re
import pandas as pd
from typing import List
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import os
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
logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    for c in existing_columns:
        try:
            column_data = df[c]
            NaN_column = column_data.isna()
            if NaN_column.any():
                logger.critical(f"Feature columns contain NaN values: column = {c}, {NaN_column.sum()} elements")
                bad_df = True

                # Get indices of rows with NaN in this column
                nan_indices = NaN_column[NaN_column].index
                logger.debug(f"NaN indices for column {c}: {list(nan_indices)}")

                # Print the entire rows for these bad indices
                for i in nan_indices:
                    logger.debug(f"{i}: {df.loc[i]}")

        except KeyError as e:
            logger.critical(f"DF is missing expected column: {c}, error: {e}")
            bad_df = True

    if bad_df:
            raise ValueError("The DataFrame is not suitable for ML model fitting. Please address these before proceeding.")
    return not bad_df
