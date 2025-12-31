import csv
from scripts.poc_extract_credit_factors import run_expectation_only_qa
from pathlib import Path


def test_run_expectation_and_review_flag():
    # run QA (writes CSVs)
    run_expectation_only_qa()
    p = Path('data/poc_qa_ambiguous.csv')
    assert p.exists(), 'poc_qa_ambiguous.csv not written'
    found_any = False
    with open(p) as fh:
        r = csv.DictReader(fh)
        for row in r:
            # ensure column present
            assert 'review_needed' in row
            if row['review_needed'].strip().lower() in ('true','1','yes'):
                found_any = True
    assert found_any, 'Expected at least one review_needed row in poc_qa_ambiguous.csv'
