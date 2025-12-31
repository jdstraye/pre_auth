# pre_auth
Pre-Authorization predicter for ShiFi/Shiloh Finance.


## Tests — Expected Warning Suppression

The test suite intentionally suppresses several recurring warnings that are expected
in our test environment and are not actionable. These suppressions keep test output
clean and focused on actual failures. The relevant configuration lives in:

- `tests/conftest.py` — Adds runtime `warnings.filterwarnings(...)` calls and sets the default
	logging level to `INFO` to reduce test noise.
- `pytest.ini` — Centralizes suppression rules so pytest consistently filters these warnings
	across environments.

The suppressed warnings include (but are not limited to):

- `UserWarning: X does not have valid feature names` from `sklearn` — occurs when a
	`DataFrame` gets converted to a `numpy` array and scikit-learn warns that the
	feature names are not attached. Tests that rely on feature names still pass and
	validate expected behavior.
- Deprecation warnings from SciPy (like `disp`/`iprint`) and various third-party
	libraries about positional args — these come from dependencies and don't affect
	our core logic.
- `Skipping check_estimators_overwrite_params` — produced by scikit-learn's internal
	estimator compliance checks; safe to ignore in our test suite.

If you want to debug or investigate these warnings, remove or adjust the filters:

1. For pytest-wide changes, edit `pytest.ini` and remove the `filterwarnings` entries.
2. For session-specific changes, remove the calls in `tests/conftest.py`.

Note: The codebase also includes defensive handling (e.g. in `eval_algos.py`) for
some cases where warnings previously arose (like single-class `predict_proba()` results).
