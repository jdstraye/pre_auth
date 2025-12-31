# Code Review Notes (starter)

This document is intended as a living line-by-line review, starting with `src/ingest.py`, `src/pipeline_coordinator.py`, and `scripts/run_exhaustive_search.py`. It lists important design choices, why they exist, and potential improvements.

- `src/ingest.py`: robust feature engineering for offers, several sentinel values (-1, -999). Consider centralizing sentinel constants and documenting their meaning.
- `src/pipeline_coordinator.py`: good separation of concerns; consider returning a richer per-fold debug artifact for deeper analysis.

To be expanded over time.
