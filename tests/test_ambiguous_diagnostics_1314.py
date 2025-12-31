import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_diagnostics_json_exists():
    p = ROOT / 'data' / 'poc_ambiguous_diagnostics_user_1314.json'
    assert p.exists(), f"Diagnostics JSON missing: {p}"


def test_three_label_crops():
    d = ROOT / 'data' / 'label_crops' / 'user_1314'
    files = list(d.glob('*.png'))
    assert len(files) >= 3, f'Expected >=3 label crops, found {len(files)}'


def test_ambiguous_count_is_three():
    p = ROOT / 'data' / 'poc_qa_ambiguous.csv'
    amb = 0
    with open(p) as fh:
        r = csv.DictReader(fh)
        for row in r:
            if row['uncertain'].strip().lower() in ('true','1','yes'):
                amb += 1
    assert amb == 3, f'Expected 3 ambiguous rows, found {amb}'
