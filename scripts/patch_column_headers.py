
"""
IMPORTANT: For feature selection, use_* keys (e.g., use_KNN, use_XGB) take precedence over "X". If a feature has "X": false but use_KNN: true, it will be included for KNN models. Always use use_* keys for model-specific inclusion/exclusion.

Script to incrementally patch src/column_headers.json to remove redundant features and add merged columns.

Steps:
1. Backup column_headers.json to column_headers.json.bak
2. Remove all *_DebtToIncome columns except DebtToIncome
3. Remove all *_Status OHE columns (e.g., AutomaticFinancing_Status_Approved, etc.)
4. Add Merged_DebtToIncome as a canonical feature
5. Review after each step; restore from backup if needed

Usage:
    python scripts/patch_column_headers.py --step <step>

Each step is idempotent and can be run/tested independently.
"""

import json
import shutil
import argparse
from pathlib import Path

SRC = Path(__file__).parent.parent / 'src' / 'column_headers.json'
BAK = SRC.with_suffix('.json.bak')

def backup():
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"Backup created: {BAK}")
    else:
        print(f"Backup already exists: {BAK}")

def restore():
    shutil.copy2(BAK, SRC)
    print(f"Restored {SRC} from backup.")

def load():
    with open(SRC, 'r') as f:
        return json.load(f)

def save(data):
    with open(SRC, 'w') as f:
        json.dump(data, f, indent=2)


def set_x_false_debttoincome(data):
    # Set 'X': 'False' for all *_DebtToIncome except DebtToIncome and Merged_DebtToIncome
    for col in data:
        if col['name'].endswith('_DebtToIncome') and col['name'] not in ['DebtToIncome', 'Merged_DebtToIncome']:
            col['X'] = 'False'
    return data

def set_x_false_ohe(data):
    # Set 'X': 'False' for all *_Status_* and *_Details_* OHE columns (not the canonical *_Status' or *_Details)
    import re
    for col in data:
        if re.match(r'.*_Status_.*', col['name']) or re.match(r'.*_Details_.*', col['name']):
            col['X'] = 'False'
    return data

def add_merged_dti(data):
    # Add Merged_DebtToIncome if not present
    if not any(col['name'] == 'Merged_DebtToIncome' for col in data):
        data.append({
            "name": "Merged_DebtToIncome",
            "categorical": "False",
            "X": "True",
            "Y": "False"
        })
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True, choices=['backup', 'restore', 'set_x_false_dti', 'set_x_false_ohe', 'add_merged_dti'])
    args = parser.parse_args()

    if args.step == 'backup':
        backup()
    elif args.step == 'restore':
        restore()
    else:
        data = load()
        if args.step == 'set_x_false_dti':
            data = set_x_false_debttoincome(data)
        elif args.step == 'set_x_false_ohe':
            data = set_x_false_ohe(data)
        elif args.step == 'add_merged_dti':
            data = add_merged_dti(data)
        save(data)
        print(f"Step {args.step} complete. Review {SRC}.")