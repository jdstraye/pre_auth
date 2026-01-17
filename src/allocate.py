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
    parser.add_argument("--strategy", choices=["stratify_labels", "time", "bin_high_ks"], default="stratify_labels", help="Split strategy to use")
    parser.add_argument("--time_col", type=str, default=None, help="Column name to use for time-based split (if omitted, will try to auto-detect *_created_at)")
    parser.add_argument("--ks_threshold", type=float, default=0.20, help="Threshold for selecting high KS features when using bin_high_ks")
    parser.add_argument("--n_bins", type=int, default=5, help="Number of quantile bins to use for bin_high_ks stratification")
    parser.add_argument("--ks_out", type=str, default=None, help="Path to write KS summary (csv)")
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

    # Choose splitting strategy
    strategy = args.strategy
    test_fraction = 0.25

    def contiguous_tail_split(d, test_fraction):
        n = len(d)
        test_n = max(100, int(test_fraction * n))
        train_df = d.iloc[:-test_n].reset_index(drop=True)
        test_df = d.iloc[-test_n:].reset_index(drop=True)
        return train_df, test_df

    if strategy == 'stratify_labels':
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
            test_size=test_fraction,
            stratify=df["stratify_key"],
            random_state=42,
        )

    elif strategy == 'time':
        # time-based contiguous tail split
        time_col = args.time_col if args.time_col else None
        if not time_col:
            candidates = [c for c in df.columns if c.endswith('_created_at') or 'date' in c.lower() or 'time' in c.lower()]
            time_col = candidates[0] if candidates else None
        if time_col and time_col in df.columns:
            # ensure parsed datetimes
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df_sorted = df.sort_values(by=time_col)
            train_df, test_df = contiguous_tail_split(df_sorted, test_fraction)
        else:
            # fallback to contiguous tail on original ordering
            train_df, test_df = contiguous_tail_split(df, test_fraction)

    elif strategy == 'bin_high_ks':
        # baseline contiguous split to estimate KS
        tr0, te0 = contiguous_tail_split(df, test_fraction)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        from scipy.stats import ks_2samp
        ks_vals = {}
        for c in num_cols:
            try:
                ks_vals[c] = float(ks_2samp(tr0[c].dropna(), te0[c].dropna()).statistic)
            except Exception:
                ks_vals[c] = 0.0
        # select high KS columns
        high_ks = [k for k, v in sorted(ks_vals.items(), key=lambda it: -it[1]) if v >= args.ks_threshold][:10]
        if not high_ks:
            high_ks = [k for k, v in sorted(ks_vals.items(), key=lambda it: -it[1])][:5]

        # create binned features
        for c in high_ks:
            bname = f"{c}__bin"
            try:
                df[bname] = pd.qcut(df[c].rank(method='first'), q=args.n_bins, labels=False, duplicates='drop').astype(str)
            except Exception:
                df[bname] = pd.cut(df[c], bins=args.n_bins, labels=False).astype(str)

        df['strat_key'] = df["final_contract_status_label"].astype(str) + "_" + df[[f"{c}__bin" for c in high_ks]].astype(str).agg('__'.join, axis=1)

        # handle tiny strata by collapsing
        counts = df['strat_key'].value_counts()
        test_n = max(100, int(test_fraction * len(df)))
        while counts.shape[0] > test_n:
            thresh = counts.min()
            rare_keys = counts[counts <= thresh].index.tolist()
            if not rare_keys:
                break
            df.loc[df['strat_key'].isin(rare_keys), 'strat_key'] = 'RARE'
            counts = df['strat_key'].value_counts()
            if counts.shape[0] <= 1:
                break

        train_df, test_df = train_test_split(df, test_size=test_fraction, stratify=df['strat_key'], random_state=42)

    else:
        # fallback
        train_df, test_df = contiguous_tail_split(df, test_fraction)

    # Drop the helper column(s) if present
    for c in ['stratify_key'] + [c for c in train_df.columns if c.endswith('__bin')]:
        if c in train_df.columns:
            try:
                train_df = train_df.drop(columns=[c])
            except Exception:
                pass
        if c in test_df.columns:
            try:
                test_df = test_df.drop(columns=[c])
            except Exception:
                pass

    logger.info("Train size: %d, Test size: %d", len(train_df), len(test_df))

    # Compute KS statistics for numeric features and optionally save
    def compute_ks(a_df, b_df):
        from scipy.stats import ks_2samp
        num_cols = [col for col in a_df.columns if pd.api.types.is_numeric_dtype(a_df[col])]
        ks = {}
        for col in num_cols:
            try:
                ks[col] = float(ks_2samp(a_df[col].dropna(), b_df[col].dropna()).statistic)
            except Exception:
                ks[col] = None
        return ks

    if args.ks_out:
        ks = compute_ks(train_df, test_df)
        ks_series = pd.Series(ks).dropna()
        summary = {
            'mean_ks': float(ks_series.mean()) if not ks_series.empty else None,
            'median_ks': float(ks_series.median()) if not ks_series.empty else None,
            'max_ks': float(ks_series.max()) if not ks_series.empty else None,
            'n_numeric': int(len(ks_series))
        }
        outp = Path(args.ks_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(ks).to_csv(outp.parent / 'ks_per_feature.csv')
        pd.DataFrame([summary]).to_csv(outp, index=False)
        logger.info("Wrote KS summary to %s", outp)

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