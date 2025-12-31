# Validation Plan

Levels:
- Unit tests: existing pytest suite (fast)
- Integration/smoke tests: ingest -> allocate -> coordinator on small data
- Regression: maintain `docs/results_history.csv` and confirm new runs do not decrease top F1 unexpectedly

Acceptance Criteria:
- All unit tests pass.
- Smoke-run on `data/prefi_weaviate_clean-2.json` completes (dry-run first).
- `generate_results_report.py` produces an entry in `docs/results_history.csv`.
