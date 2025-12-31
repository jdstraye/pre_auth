import tempfile
import pandas as pd
from pathlib import Path

from scripts.generate_results_report import main


def test_generate_report_creates_files(tmp_path):
    md = tmp_path / 'models' / 'intensive_search'
    md.mkdir(parents=True)
    # create a fake search_results.csv with a mean_f1
    df = pd.DataFrame([{'mean_f1': 0.77, 'model': 'RandomForestClassifier', 'params': '{}'}])
    df.to_csv(md / 'search_results.csv', index=False)

    # run generator
    main(str(md))

    # check reports and history
    reports = Path('docs/reports')
    history = Path('docs/results_history.csv')
    assert reports.exists()
    assert history.exists()
