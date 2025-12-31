# Developer Guide

Quick start:
1. Create and activate venv: `source .venv_pre_auth/bin/activate`.
2. Install dev deps: `pip install -r requirements.txt`.
3. Run unit tests: `pytest -q`.
4. Run a dry-run of an intensive search: `python scripts/run_intensive_search.py --dry-run`.

Key components:
- `src/ingest.py`: JSON -> preprocessed DataFrame, one-hot & ordinal mapping.
- `src/pipeline_coordinator.py`: builds pipelines with `MaybeSMOTESampler` and `FeatureSelectingClassifier` and exposes `search_models` and `fit_pipeline`.
- `scripts/run_exhaustive_search.py`: heavy combinatorial search harness.
- `scripts/run_intensive_search.py`: wrapper with denser defaults.

How to contribute:
- Write focused unit tests for new behavior.
- Update docs in `docs/` and append a `docs/reports/report_{ts}.md` when adding model improvements.
