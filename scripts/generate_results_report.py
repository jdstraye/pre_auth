#!/usr/bin/env python3
"""Generate a Markdown report from search results and update history/plot."""
from pathlib import Path
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORTS = DOCS / "reports"
PLOTS = DOCS / "plots"
REPORTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def find_latest_results(models_dir: Path):
    # look for search_results.csv or top_candidates.json
    csv = models_dir / 'search_results.csv'
    if csv.exists():
        df = pd.read_csv(csv)
        if not df.empty:
            return df.sort_values(by='mean_f1', ascending=False).iloc[0].to_dict()
    jsonf = models_dir / 'top_candidates.json'
    if jsonf.exists():
        obj = json.loads(jsonf.read_text())
        if obj:
            return obj[0]
    return None


def append_history(date: str, f1: float, model: str, params: str):
    history = DOCS / 'results_history.csv'
    row = pd.DataFrame([{'date': date, 'best_mean_f1': f1, 'model': model, 'params': params}])
    if history.exists():
        row.to_csv(history, mode='a', header=False, index=False)
    else:
        row.to_csv(history, index=False)


def plot_history():
    hist = DOCS / 'results_history.csv'
    if not hist.exists():
        return
    df = pd.read_csv(hist, parse_dates=['date'])
    df = df.sort_values('date')
    plt.figure()
    plt.plot(df['date'], df['best_mean_f1'], marker='o')
    plt.ylabel('Best mean F1')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS / 'f1_over_time.png')


def main(models_dir: str = 'models/intensive_search'):
    md = find_latest_results(Path(models_dir))
    ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    if md is None:
        print('No results found in', models_dir)
        return
    # prepare report
    report = REPORTS / f'report_{ts}.md'
    with open(report, 'w', encoding='utf8') as f:
        f.write(f"# Intensive Search Report {ts}\n\n")
        f.write(f"- model: {md.get('model')}\n")
        f.write(f"- mean_f1: {md.get('mean_f1')}\n")
        f.write(f"- params: {md.get('params')}\n")
    append_history(ts, float(md.get('mean_f1', 0.0)), md.get('model', ''), json.dumps(md.get('params', {})))
    plot_history()
    print('Report generated:', report)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', default='models/intensive_search')
    args = p.parse_args()
    main(args.models_dir)
