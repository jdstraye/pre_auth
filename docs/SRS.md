# Software Requirements Specification (SRS)

Overview
- Purpose: Find a classifier achieving >90% macro F1 on `final_contract_status_label` (status) and provide per-candidate metrics and feature importances.
- Inputs: preprocessed CSVs (from `src/ingest.py`) and raw JSON (`data/prefi_weaviate_clean-2.json`).
- Outputs: trained pipelines, per-candidate metrics (accuracy, precision, recall, f1), feature importances, reports in `docs/reports/`.

Functional Requirements
- Ingest and preprocess raw JSON to canonical CSVs.
- Allocate train/test datasets.
- Run intensive hyperparameter search over multiple classifiers and hyperparameter distributions.
- Record per-candidate cross-validated metrics and selected features/importance scores.

Non-functional Requirements
- Reproducibility (random seeds controlled via `gv.RANDOM_STATE`).
- Scalable search (support for distributed/cluster execution later).
