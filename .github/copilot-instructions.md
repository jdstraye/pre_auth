# Copilot / AI contributor instructions — pre_auth

Purpose: give an AI coding agent the minimal, high-value knowledge to be productive in this repo in minutes (not hours). Keep edits small, well-tested and explain behavioral changes with concrete test cases.

## Quick orientation (big picture)
- Goal: extract structured credit-factors from PDF credit summaries and train/evaluate ML classifiers (status/tier). Key flows: PDF → span/color/layout extraction → canonical factorization → feature pipeline → model training/eval.
- Major components:
  - `src/scripts/pdf_color_extraction.py` — canonical PDF extractor (primary implementation; prefer this for changes).
  - `scripts/poc_extract_credit_factors.py` — legacy POC (kept for diagnostics). Do not change callers to import this unless explicitly preserving backward compatibility.
  - `src/components/smote_sampler.py` — name-aware SMOTE wrapper and SMOTE-related safeguards used across training pipelines.
  - `scripts/run_sample_extraction.py`, `scripts/validate_credit_factors.py`, `scripts/eval_color_extractor.py` — utilities to run and validate extraction locally.
  - `data/extracted/`, `data/pdf_analysis/`, `data/label_crops/` — canonical fixtures, diagnostics and labeled crops used by tests and debugging.

## Important project-specific conventions (read before coding)
- Canonical-first: prefer `src/scripts/pdf_color_extraction.py` over legacy POC. Add compatibility shims only when absolutely necessary.
- Color sampling: prefer `span`-based color detection (`combined_sample_color_for_phrase` / `span_color_hex`) before any pixel-sampling fallback (e.g., `median_5x5`). Tests assert span-first behavior.
- Deterministic & strict GTs: tests assert exact ground-truth values (do not relax comparisons). If behaviour must change, add/update GT fixtures in `data/extracted/` and a deterministic unit test.
- SMOTE by name: pipelines use `MaybeSMOTESampler` / `NamedSMOTE(NAMEDSMOTENC)` — pass categorical feature names (keys use `smote__...`). Avoid changing API shape; add tests for sampling behavior when modifying.
- Config-driven debug/flow: `config/control.ini` (section `[poc]`) controls debugging/marker behavior (`marker_mode`, `debug_phrase`). Use it for reproducible debugging runs.
- No large generated assets in code commits: images in `data/poc_imgs/` and `data/label_crops/` are large — prefer keeping them out of feature branches or use `git-lfs`.

## Where to start (high-value entry points & examples)
- Reproduce an extraction for a single user (fast):
  - `python scripts/run_sample_extraction.py user_1314` — writes `data/extracted/...` (used by many tests).
- Quick color-sampling check (REPL):
  - >>> from src.pymupdf_compat import fitz
  - >>> from src.scripts.pdf_color_extraction import combined_sample_color_for_phrase
  - >>> doc = fitz.open('data/pdf_analysis/user_1314.pdf')
  - >>> combined_sample_color_for_phrase(doc, 'Drop Bad Auth', page_limit=2)
- Run the focused regression that must remain green when changing extractor:
  - `pytest -q tests/test_poc_fixes.py::test_user_1314_drop_bad_auth_preserved`
- Run the full suite locally (CI gate):
  - Ensure `.venv_pre_auth` active; then `pytest -q` (expect: ~260 passed locally).

  IMPORTANT: Always activate the repository virtualenv before running PDF extraction or any PyMuPDF-dependent commands.
  - In Linux/macOS: run `source .venv_pre_auth/bin/activate` (or `. .venv_pre_auth/bin/activate`).
  - For one-off commands without activation, run: `.venv_pre_auth/bin/python scripts/<script>.py`.
  - Quick checks after activation: `which python` should point to `.venv_pre_auth/bin/python`, and `python -c "import fitz; print('ok')"` should succeed.

  AI agents and contributors: DO NOT run PDF extraction or any code that imports `fitz` unless the venv is active and the `fitz` import succeeds. Make the activation step explicit in any shell commands you execute.

## Debugging checklist (common root causes & where to look)
- Missing/incorrect summary line in extraction:
  - Inspect `data/extracted/<user>_credit_summary*.json`, `data/pdf_analysis/<user>*.json`, and `data/label_crops/<user>`.
  - In code: check contiguous-span exact-match logic in `pdf_color_extraction.py` and the deterministic merge/preservation logic.
- SMOTE failures or flaky CV:
  - Inspect `src/components/smote_sampler.py` and use the debug helpers in `src/debug_library.py`.
  - Check `tests/test_parameter_sampling.py` and `tests/test_smote_fallback.py` for expected behavior.
- PyMuPDF Swig warnings:
  - Use `from src.pymupdf_compat import fitz` in tests/scripts to avoid noisy, fragile imports.

## Testing & CI expectations (do this for any meaningful change)
- Add a focused unit test reproducing the observed bug/regression (place under `tests/` with an explanatory name).
- Add or update a GT fixture in `data/extracted/` for behavioral changes — tests must reference that fixture.
- Run focused tests first, then full suite locally before pushing:
  - Focused: `pytest -q tests/test_poc_fixes.py::test_user_1314_drop_bad_auth_preserved`
  - Subset: `pytest -q tests/test_poc_fixes.py tests/test_poc_primary_adoption.py`
  - Full: `pytest -q` (CI should be green before merging).

## PR & commit guidance (project norms)
- Keep commits atomic and scoped: `feat(pdf): ...`, `fix(smote): ...`, `test(poc): ...`.
- Use `git add -p` to avoid bundling unrelated changes (especially avoid committing generated images).
- Explain ground-truth updates in the PR body and include before/after examples (user id + snippet).
- If touching extraction heuristics, include regression fixtures and a short note in `TODO.md` if further follow-up is required.

## Useful files to inspect for context
- Extraction: `src/scripts/pdf_color_extraction.py`, `scripts/poc_extract_credit_factors.py` (POC)
- SMOTE / pipelines: `src/components/smote_sampler.py`, `src/pipeline_coordinator.py`, `src/eval_algos.py`
- Tests & GTs: `tests/`, `data/extracted/`, `data/pdf_analysis/`, `tests/test_poc_fixes.py`
- Config & dev: `config/control.ini` (POC flags), `requirements.txt`, `.venv_pre_auth/` (local venv)
- Docs/TODO: `README.md`, `TODO.md`, `doc/preprocessing.md`

## Design Rules:
1. Do not use token matching to identify color. The extraction must be generic and work for thousands of cases.
2. Do not use environment variables for flow control. Environment variables should only be used for secrets. Flow control variables should be set in the config/control.ini file.
3. Before delivering code, ensure it adheres to PEP 8 standards for Python code.
4. Write unit tests for any new functionality added. Ensure that all existing and new unit tests pass before delivering code.
    a) Use pytest for writing and running tests.
    b) Aim for at least 80% code coverage on new modules on the first delivery.
    c) Include tests for edge cases and error handling.
    d) Document the tests clearly, explaining what each test is verifying.
    e) Organize tests in a separate `tests/` directory, mirroring the structure of the `src/` directory.
5. When updating documentation, ensure that all relevant sections are updated to reflect the changes.
6. In addition to unit tests, write integration tests for any new modules that interact with existing components.
    a) Ensure that integration tests cover the interaction between new and existing modules.
    b) Use realistic data scenarios in integration tests to simulate real-world usage.
    c) Document integration tests clearly, explaining the purpose and expected outcomes.
    d) Starting integration tests are tests/test_pdf_extraction_ground_truth.py run with cli arguments for users whose ground truths are validated in data/extracted, as:
    - `pytest tests/test_pdf_extraction_ground_truth.py --user_id 582` because data/extracted/user_582_credit_summary_*ground_truth.json exists (validated grround truth file)