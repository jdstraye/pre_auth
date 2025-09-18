"""
Single record prediction script for pre-authorization decisions.
Located at src/predict.py
"""

import json
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, cast
import warnings
import os
from src.for_build import SelectKBestDf, NamedSMOTENC
import xgboost
from xgboost import XGBClassifier



warnings.filterwarnings("ignore")

# Absolute imports since src is a package
from src import ingest, utils
from src.utils import logger


def resource_path(relative_path: str) -> Path:
    """
    Resolve resource paths for both development and PyInstaller exe.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running inside PyInstaller bundle
        base_path = Path(getattr(sys, "_MEIPASS"))
    else:
        # Running in normal Python environment
        base_path = Path(__file__).parent.parent
    return base_path / relative_path


class PreAuthPredictor:
    """Pre-authorization prediction engine."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        schema_path: Optional[Path] = None,
    ):
        # Resolve defaults if not provided
        self.models_dir = (
            models_dir if models_dir is not None else resource_path("models")
        )
        self.schema_path = (
            schema_path
            if schema_path is not None
            else resource_path("src/column_headers.json")
        )

        # Initialize placeholders with their correct types for mypy/Pylance
        self.status_model: Any = None
        self.tier_model: Any = None
        self.schema: List[Dict[str, Any]] = []
        self.sorted_schema: List[Dict[str, Any]] = []
        self.column_map: Dict[str, Dict[str, Any]] = {}
        self.feature_cols: list[str] = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load models and schema
        self._load_models_and_schema()

    def _load_models_and_schema(self) -> None:
        """Load trained models and schema configuration."""

        # Load models
        status_model_path = self.models_dir / "status_best.pkl"
        tier_model_path = self.models_dir / "tier_best.pkl"

        if not status_model_path.exists():
            raise FileNotFoundError(f"Status model not found at {status_model_path}")
        if not tier_model_path.exists():
            raise FileNotFoundError(f"Tier model not found at {tier_model_path}")

        # Cast to Any to satisfy Pylance, as joblib.load returns a generic object
        self.status_model = cast(Any, joblib.load(status_model_path))
        self.tier_model = cast(Any, joblib.load(tier_model_path))
        self.logger.info("Successfully loaded trained models")

        # Load schema using the ingest module
        self.schema = ingest._load_golden_schema(self.schema_path)
        self.sorted_schema, self.column_map = ingest._parse_schema(self.schema)

        # Extract feature columns and sanitize names
        self.feature_cols = [
            utils.sanitize_column_name(col['name']) 
            for col in self.schema 
            if col.get('X') == 'True'
        ]
        self.logger.info(f"Loaded schema with {len(self.feature_cols)} feature columns")

    def _preprocess_single_record(self, json_path: Path) -> pd.DataFrame:
        """Preprocess a JSON file with a single record into a feature-ready DataFrame."""
        try:
            df = ingest.flatten_weaviate_data(json_path, self.sorted_schema)
            final_df = ingest.preprocess_dataframe(
                df, self.sorted_schema, self.column_map
            )
            return final_df
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

    def predict_single_record(self, json_path: Path) -> Dict[str, Any]:
        """Run the complete prediction pipeline on a single record in a JSON file."""
        with open(json_path, "r") as f:
            json_data = json.load(f)

        processed_df = self._preprocess_single_record(json_path)

        # Status prediction
        X_features = processed_df[self.feature_cols]

        # Type guard to ensure models are not None
        if not self.status_model:
            raise RuntimeError("Status model is not loaded.")

        # `predict` is a known attribute of the loaded model object
        status_pred = self.status_model.predict(X_features)[0]
        status_probs = self.status_model.predict_proba(X_features)[0]

        results: Dict[str, Any] = {
            "status_prediction": {
                "predicted_class": "approved" if status_pred == 0 else "declined",
                "probabilities": {
                    "approved": float(status_probs[0]),
                    "declined": float(status_probs[1]),
                },
            }
        }

        # Tier prediction if approved
        if status_pred == 0:
            if not self.tier_model:
                raise RuntimeError("Tier model is not loaded.")
            
            tier_pred = self.tier_model.predict(X_features)[0]
            tier_probs = self.tier_model.predict_proba(X_features)[0]

            tier_labels = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
            results["tier_prediction"] = {
                "predicted_tier": tier_labels[tier_pred]
                if tier_pred < len(tier_labels)
                else f"Tier {tier_pred}",
                "probabilities": {
                    tier_labels[i] if i < len(tier_labels) else f"Tier {i}": float(prob)
                    for i, prob in enumerate(tier_probs)
                },
            }

        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict pre-authorization status and tier"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON file, the data")
    parser.add_argument("--models_dir", type=Path, help="Directory of trained models")
    parser.add_argument("--schema", type=Path, help="Schema JSON path, e.g., src/column_headers.json")
    parser.add_argument("--output", type=Path, help="Optional output JSON file")

    args = parser.parse_args()

    predictor = PreAuthPredictor(models_dir=args.models_dir, schema_path=args.schema)
    input_path = Path(args.input)
    logger.debug(f"{input_path = }")
    results = predictor.predict_single_record(input_path)

    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)

if __name__ == "__main__":
    main()
