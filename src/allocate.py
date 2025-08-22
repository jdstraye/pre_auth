"""
Split the source data into 75% training and 25% testing while keeping 
the proportions of data similar. Use stratified sampling to ensure 
proportionality.
e.g., if 10% of the testing are rejected, then 10% of training should 
also be rejected.
Columns in particular - final_contract_status_label and final_contract_tier_label

Usage:
    python src/allocate.py
"""

import os
import shutil
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load your CSV file
input_path = Path("data/prefi_weaviate_clean-1_processed.csv")

df: DataFrame = pd.read_csv(input_path)
logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])

# Make sure the required columns exist
required_cols = ["final_contract_status_label", "final_contract_tier_label"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Create stratification key by combining the two columns
df["stratify_key"] = df["final_contract_status_label"].astype(str) + "_" + df["final_contract_tier_label"].astype(str)

# For stratification, need to duplicate some rare rows to make the test and train data sets balanced -
counts = df['stratify_key'].value_counts()
if (counts < 2).any():
    logging.debug(f"Before balancing: {df['stratify_key'].value_counts() = }")
    rare = counts[counts < 2].index
    df_balanced = pd.concat([
        df,
        df[df['stratify_key'].isin(rare)]  # duplicate the rare rows once
    ])
    df=df_balanced
    logging.debug(f"After balancing: {df['stratify_key'].value_counts() = }")

# Split train/test
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    stratify=df["stratify_key"],
    random_state=42,
)

# Drop the helper column
train_df = train_df.drop(columns=["stratify_key"])
test_df = test_df.drop(columns=["stratify_key"])

logger.info("Train size: %d, Test size: %d", len(train_df), len(test_df))

# Save the splits
output_dir = Path(__file__).parent.parent / "data" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M")
train_path = output_dir / f"train_{timestamp}.csv"
test_path = output_dir / f"test_{timestamp}.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

logger.info("Saved train split to %s", train_path)
logger.info("Saved test split to %s", test_path)

# Also maintain "latest" symlinks
latest_train = output_dir / "train_latest.csv"
latest_test = output_dir / "test_latest.csv"

if latest_train.exists() or latest_train.is_symlink():
    latest_train.unlink()
if latest_test.exists() or latest_test.is_symlink():
    latest_test.unlink()

latest_train.symlink_to(train_path.absolute())
latest_test.symlink_to(test_path.absolute())

def make_symlink_or_copy(src: Path, dest: Path):
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(src, dest)
        logger.info("Symlinked %s → %s", dest, src)
    except (AttributeError, NotImplementedError, OSError):
        # Fallback: copy file if symlink is not supported (e.g. on Windows without admin rights)
        shutil.copy2(src, dest)
        logger.info("Copied %s → %s", src, dest)

make_symlink_or_copy(train_path, latest_train)
make_symlink_or_copy(test_path, latest_test)