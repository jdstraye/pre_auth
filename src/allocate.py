"""
Split the source data into 75% training and 25% testing while keeping
the proportions of data similar. Use stratified sampling to ensure
proportionality.
e.g., if 10% of the testing are rejected, then 10% of training should
also be rejected.
Columns in particular - final_contract_status_label and final_contract_tier_label
Usage:
    python src/allocate.py -i data/input.csv -o1 data/train.csv -o2 data/test.csv -d -l
"""
import sys
import os
import shutil
import argparse
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Split data into train/test sets.")
    parser.add_argument("-i", "--input_csv", required=True, help="Input CSV file path")
    parser.add_argument("-o1", "--out_train", required=True, help="Output train CSV file path")
    parser.add_argument("-o2", "--out_test", required=True, help="Output test CSV file path")
    parser.add_argument("-d", "--add_date", action="store_true", help="Append date/time to output filenames")
    parser.add_argument("-l", "--create_links", action="store_true", help="Create latest symlinks")
    return parser.parse_args()

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

def main():
    args = parse_args()

    # Load your CSV file
    input_path = Path(args.input_csv)
    # When the CSV contains sentinel strings like 'NA' we must not coerce
    # them to NaN so downstream validators see the intended sentinel values.
    df: DataFrame = pd.read_csv(input_path, keep_default_na=False)
    logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])

    # Make sure the required columns exist
    required_cols = ["final_contract_status_label", "final_contract_tier_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create stratification key by combining the two columns
    df["stratify_key"] = df["final_contract_status_label"].astype(str) + "_" + df["final_contract_tier_label"].astype(str)

    # For stratification, need to duplicate some rare rows to make the test and train data sets balanced
    counts = df['stratify_key'].value_counts()
    if (counts < 2).any():
        logging.debug(f"Before balancing: {df['stratify_key'].value_counts() = }")
        rare = counts[counts < 2].index
        df_balanced = pd.concat([
            df,
            df[df['stratify_key'].isin(rare)]  # duplicate the rare rows once
        ])
        df = df_balanced
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
    output_dir = Path(args.out_train).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided output paths or append date/time if --add_date is set
    if args.add_date:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        train_path = output_dir / f"{Path(args.out_train).stem}_{timestamp}.csv"
        test_path = output_dir / f"{Path(args.out_test).stem}_{timestamp}.csv"
    else:
        train_path = Path(args.out_train)
        test_path = Path(args.out_test)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved train split to %s", train_path)
    logger.info("Saved test split to %s", test_path)

    # Create latest symlinks if --create_links is set
    if args.create_links:
        latest_train = output_dir / "train_latest.csv"
        latest_test = output_dir / "test_latest.csv"
        if latest_train.exists() or latest_train.is_symlink():
            latest_train.unlink()
        if latest_test.exists() or latest_test.is_symlink():
            latest_test.unlink()
        make_symlink_or_copy(train_path.absolute(), latest_train)
        make_symlink_or_copy(test_path.absolute(), latest_test)

if __name__ == "__main__":
    main()
    sys.exit(0)