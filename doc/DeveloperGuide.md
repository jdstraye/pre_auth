# Developer Guide — pre_auth

## Feature Selection Precedence

**IMPORTANT:**
For feature selection, use_* keys (e.g., use_KNN, use_XGB) take precedence over "X". If a feature has "X": false but use_KNN: true, it will be included for KNN models. Always use use_* keys for model-specific inclusion/exclusion.

## Quick Start

1. **Clone the repository** and set up your Python environment:
   - Use the provided `.venv_pre_auth` virtual environment.
   - Activate with `source .venv_pre_auth/bin/activate`.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Run a sample extraction**:
   - `python scripts/run_sample_extraction.py user_1314`
   - Output: `data/extracted/...` (used by tests).

3. **Run tests**:
   - Focused regression: `pytest -q tests/test_poc_fixes.py::test_user_1314_drop_bad_auth_preserved`
   - Full suite: `pytest -q`

4. **Debugging**:
   - Use `config/control.ini` to control debug/marker behavior.
   - Inspect output in `data/extracted/`, `data/pdf_analysis/`, and `data/label_crops/`.

## Project Overview

- **Goal:** Extract structured credit-factors from PDF credit summaries and train/evaluate ML classifiers (status/tier).
- **Key Flows:** PDF → span/color/layout extraction → canonical factorization → feature pipeline → model training/eval.
- **Major Components:**
  - `src/scripts/pdf_color_extraction.py`: Canonical PDF extractor.
  - `src/components/smote_sampler.py`: SMOTE wrapper for training pipelines.
  - `scripts/run_sample_extraction.py`, `scripts/validate_credit_factors.py`: Utilities for extraction/validation.
  - `data/extracted/`, `data/pdf_analysis/`, `data/label_crops/`: Fixtures and diagnostics.

## Conventions & Best Practices

- **Canonical-first:** Prefer canonical implementations over legacy POC scripts.
- **Color sampling:** Use span-based color detection before pixel-sampling fallback.
- **Deterministic ground truths:** Tests assert exact GT values; update fixtures and add tests for changes.
- **SMOTE by name:** Use `MaybeSMOTESampler` / `NamedSMOTE` and pass categorical feature names.
- **Config-driven debug/flow:** Use `config/control.ini` for reproducible debugging runs.
- **No large generated assets in commits:** Keep images out of feature branches or use `git-lfs`.
- **Testing:**
  - Add focused unit tests for bugs/regressions.
  - Update GT fixtures for behavioral changes.
  - Run focused and full test suites before pushing.
- **Commits:**
  - Keep commits atomic and scoped.
  - Use `git add -p` for clean history.
  - Explain GT updates in PRs.

## File Reference

- Extraction: `src/scripts/pdf_color_extraction.py`, `scripts/poc_extract_credit_factors.py`
- SMOTE / pipelines: `src/components/smote_sampler.py`, `src/pipeline_coordinator.py`, `src/eval_algos.py`
- Tests & GTs: `tests/`, `data/extracted/`, `data/pdf_analysis/`, `tests/test_poc_fixes.py`
- Config & dev: `config/control.ini`, `requirements.txt`, `.venv_pre_auth/`
- Docs/TODO: `README.md`, `TODO.md`, `doc/preprocessing.md`

## Troubleshooting

- **Missing/incorrect summary line:**
  - Check contiguous-span logic in `pdf_color_extraction.py` and merge/preservation logic.
- **SMOTE failures/flaky CV:**
  - Inspect `src/components/smote_sampler.py` and debug helpers in `src/debug_library.py`.
- **PyMuPDF Swig warnings:**
  - Use `from src.pymupdf_compat import fitz` in tests/scripts.

## Further Reading
- See `doc/preprocessing.md` for pipeline details.
- See `README.md` for high-level project info.
- See `TODO.md` for open issues and follow-ups.
