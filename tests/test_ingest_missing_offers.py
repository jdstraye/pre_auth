import pandas as pd
from src.ingest import flatten_weaviate_data, _load_golden_schema, _parse_schema
from pathlib import Path


def test_missing_debt_resolution_offer_mapped_to_nan():
    schema = _load_golden_schema(Path('src/column_headers.json'))
    sorted_schema, column_map = _parse_schema(schema)
    df = flatten_weaviate_data(Path('data/prefi_weaviate_clean-2.json'), sorted_schema)
    # Find the known problematic record_id
    rid = 'aac3e33d1035e1b6a03c29c7dc7d7645825c77b28a559ad36fd49ecb82fc27da'
    row = df[df['record_id'] == rid]
    assert not row.empty
    # Missing flag should be 1
    assert int(row.iloc[0]['DebtResolution_missing_']) == 1
    # numeric representation of below_600 should be NaN (not -1)
    assert pd.isna(row.iloc[0]['DebtResolution_below_600_'])
    # other numeric fields should be NaN as well
    assert pd.isna(row.iloc[0]['DebtResolution_Score']) or row.iloc[0]['DebtResolution_Score'] == -999
    assert pd.isna(row.iloc[0]['DebtResolution_DebtToIncome'])
