# Conversation Summary (20260127-<timestamp>)

## Main Topics
- Dynamic column selection for ML pipelines
- Schema-driven vs. classifier-dependent feature selection
- OHE vs. enumerated categoricals for CatBoost/XGBoost
- Test-driven validation of column logic
- Next steps: schema/pipeline update for classifier-aware feature selection

## Key Decisions
- Confirmed current pipeline only uses OHE columns as features
- Determined CatBoost/XGBoost should use enumerated categoricals, not OHE
- Plan to update schema and pipeline logic to select features dynamically based on classifier type

## Status
- All column selection tests pass (schema-driven, OHE only)
- Need to update schema and pipeline for classifier-aware feature selection
- Next: brainstorm and implement dynamic feature selection logic

## Next Steps
1. Update schema to mark enumerated categoricals as features
2. Update pipeline logic to:
   - Use enumerated categoricals for CatBoost/XGBoost
   - Use OHE for models that require numeric input
   - Dynamically select features based on classifier type
3. Add/adjust tests to validate new logic
4. Commit and push changes

---

# Conversation Log

(Full chat history to be appended here by the assistant)
