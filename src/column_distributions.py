import pandas as pd
import sys

def count_unique_values(csv_file: str) -> None:
    """Print unique value counts for each column in a CSV file."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)

    print(f"Column distributions for {csv_file}")
    print("--------------------------------")
    
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        print(f"Column: {col}")
        for value, count in counts.items():
            print(f"  {value}: {count}")
        print()

    # Highlight status and tier columns
    for target in ["final_contract_status_label", "final_contract_tier_label"]:
        if target in df.columns:
            print(f"Summary for {target}:")
            counts = df[target].value_counts(dropna=False)
            for value, count in counts.items():
                print(f"  {value}: {count}")
            print()
        else:
            print(f"Warning: {target} not found in {csv_file}")

        # Encoding
        target_base = target.replace('_label', '')
        unique_values = df[target].unique()
        for val in unique_values:
            sample_row = df[df[target] == val].iloc[0]
            print(f"Sample row for {target} = {val}:")
            print(sample_row[[target, target_base]])
    

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/prefi_weaviate_clean-1_modified_import_processed.csv"
    count_unique_values(input_csv)
    sys.exit(0)