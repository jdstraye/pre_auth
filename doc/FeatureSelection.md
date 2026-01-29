
## Feature Selection Precedence

**IMPORTANT:**

For feature selection, use_* keys (e.g., use_KNN, use_XGB) take precedence over "X". If a feature has "X": false but use_KNN: true, it will be included for KNN models. Always use use_* keys for model-specific inclusion/exclusion.

This rule applies across all pipeline, ingest, and configuration logic.