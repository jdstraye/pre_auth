# Results Tracker

This document tracks the best candidate results from intensive hyperparameter searches.

Each run saves a snapshot in `docs/reports/` named `report_{timestamp}.md` and a consolidated CSV `docs/results_history.csv` is maintained with columns:

- `date` (UTC timestamp)
- `best_mean_f1`
- `model`
- `params` (JSON)
- `selected_features` (JSON)
- `feature_importances` (JSON)

Maintainers: append the new report after every major improvement and update the `TODO.md` entry noting the change.

Plots: `docs/plots/f1_over_time.png` shows the best mean F1 by date.

## How to reproduce a run

Run the smoke dry-run (fast):

```bash
# activate your venv (example)
source .venv_pre_auth/bin/activate
python3 scripts/run_intensive_search.py --start-json data/prefi_weaviate_clean-2.json --dry-run
```

Run a limited-intensity real search (short-ish):

```bash
python3 scripts/run_exhaustive_search.py --start-json data/prefi_weaviate_clean-2.json --n-samples-per-model 50 --limit 10 --n-top 5 --output-dir models/intensive_search/limited_run_50
```

When ready, run the very intensive search (long-running, tune `--n-samples-per-model` and `--n-jobs`):

```bash
python3 scripts/run_intensive_search.py --start-json data/prefi_weaviate_clean-2.json --n-samples-per-model 2000 --n-jobs 8 --n-top 50 --output-dir models/intensive_search/very_intensive_run
```

Note: a very intensive run was started on 2025-12-15 and is writing to `models/intensive_search/very_intensive_run`.
