# Architecture

High-level overview
- Ingest -> Preprocess -> Allocate -> Search: We ingest raw JSON, preprocess to a canonical schema, allocate train/test, then run a hyperparameter search using `MLPipelineCoordinator`.

Function-by-function crawl (short):
- `src/ingest.py`: `flatten_weaviate_data`, `preprocess_dataframe`, helper cleaning functions.
- `src/pipeline_coordinator.py`: `create_pipeline`, `fit_pipeline`, `cross_validate_pipeline`, `search_models`.
- `scripts/run_exhaustive_search.py`: enumerates/ samples parameter combinations, creates per-fold pipeline runs, saves results.

Current challenges and future opportunities
- Scalability of exhaustive search; add distributed or asynchronous execution.
- Better reporting/ dashboards for results.
