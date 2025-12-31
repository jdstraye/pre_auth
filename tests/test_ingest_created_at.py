import json
from pathlib import Path
import pandas as pd
from src.ingest import flatten_weaviate_data, preprocess_dataframe, _load_golden_schema, _parse_schema


def test_contract_created_at_extracted(tmp_path):
    sample = {
        "data": [
            {
                "record_id": "r1",
                "user_initials": "AA",
                "contracts": [
                    {"contract_id": "c1", "created_at": "2024-11-19T00:00:00-05:00", "status": "approved", "amount": "1000", "tier": "0.24"}
                ],
                "prefi_data": {"Offers": []}
            }
        ]
    }
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(sample))

    # Load schema and parse
    schema = _load_golden_schema(Path('src/column_headers.json'))
    sorted_schema, column_map = _parse_schema(schema)

    df = flatten_weaviate_data(p, schema)
    assert 'final_contract_created_at' in df.columns
    # Now preprocess and check dtype
    processed = preprocess_dataframe(df, sorted_schema, column_map)
    assert 'final_contract_created_at' in processed.columns
    assert pd.api.types.is_datetime64_any_dtype(processed['final_contract_created_at'])
