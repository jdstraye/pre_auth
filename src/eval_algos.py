"""
Evaluates machine learning models for status and tier classification using SMOTE,
category grouping, and StratifiedKFold. Saves top 100 models per phase and tier target encoder.
"""
from cmath import phase
import sys
import numpy as np
import pandas as pd
import logging
import joblib
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Union, Sequence, Any, Dict, cast, TypedDict
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report   
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
from imblearn.over_sampling import SMOTENC  # type: ignore
from imblearn.pipeline import Pipeline as ImbPipeline
#2020830 from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from utils import check_df_columns, sanitize_column_name, CustomRotatingFileHandler

# Logging Setup
Path("logs").mkdir(exist_ok=True)
handler = CustomRotatingFileHandler("logs/pre_auth_eval_algos.log", maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.info("Logger initialized for eval_algos.py.")

"""
Evaluates machine learning models for status and tier classification using SMOTE,
category grouping, and StratifiedKFold. Saves top 100 models per phase and tier target encoder.
"""

# Helper Functions
#20250830 def group_rare_classes(y: Union[np.ndarray, Sequence[int]], min_samples: int = 5, phase: str = 'tier') -> Tuple[np.ndarray, Optional[LabelEncoder]]:
#20250830     """Group classes with < min_samples into 'other' for tier phase."""
#20250830     if phase != 'tier':
#20250830         return np.array(y, dtype=int), None  # No grouping for status
#20250830     class_counts = pd.Series(list(y)).value_counts()
#20250830     logger.info(f"Original {phase} class counts: {class_counts.to_dict()}")
#20250830     # Group based on sample counts: low (0,2,3,5,9,10), medium (1,4), high (6,7,8)
#20250830     group_dict = {
#20250830         0: 'low', 2: 'low', 3: 'low', 5: 'low', 9: 'low', 10: 'low',
#20250830         1: 'medium', 4: 'medium',
#20250830         6: 'high', 7: 'high', 8: 'high'
#20250830     }
#20250830     y_grouped = np.array([group_dict.get(x, 'low') for x in y])
#20250830     le = LabelEncoder()
#20250830     y_grouped = le.fit_transform(y_grouped)
#20250830     logger.info(f"Grouped {phase} classes: New unique {len(np.unique(y_grouped))}")
#20250830     y_list: List[Any] = np.array(y_grouped).tolist()
#20250830     logger.info(f"New class counts: {pd.Series(y_list).value_counts().to_dict()}")
#20250830     return np.array(y_grouped, dtype=int), le

class ParamDict(TypedDict, total=False):
    select__k: int
    smote__categorical_features: List[int]

def log_feature_importance(model, feature_names: List[str]) -> None:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if importances.ndim > 0:
            logger.info("Feature importances:")
            for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                logger.info(f"{name}: {imp:.4f}")
    else:
        logger.info("Feature importances not available for this model.")

def log_class_imbalance(y: Union[np.ndarray, Sequence[int]], phase: str) -> None:
    counts = pd.Series(list(y)).value_counts().sort_index()
    logger.info(f"{phase} class imbalance:")
    for label, count in counts.items():
        logger.info(f"Class {label}: {count} samples")
    if any(count < 5 for count in counts):
        logger.warning(f"{phase}: Some classes have fewer than 5 samples, which may cause issues with StratifiedKFold or SMOTENC.")

def load_column_headers(column_headers_json: Path, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Load feature, categorical, and target columns from column_headers.json."""
    try:
        with open(column_headers_json, 'r', encoding='utf-8') as f:
            header_data = json.load(f)
        feature_cols = [sanitize_column_name(col['name']) for col in header_data if col.get('X') == 'True']
        check_df_columns(df, feature_cols)
        categorical_cols = [col['name'] for col in header_data if col.get('categorical') == 'True']
        target_cols = [col['name'] for col in header_data if col.get('Y') == 'True']
        logger.info(f"Loaded {len(feature_cols)} feature columns, {len(categorical_cols)} categorical columns, and {len(target_cols)} target columns from column_headers.json")
        return feature_cols, categorical_cols, target_cols
    except FileNotFoundError:
        logger.error("Error: 'data/column_headers.json' not found.")
        raise
    except json.JSONDecodeError:
        logger.error("Error: Could not decode 'data/column_headers.json'. Check for syntax errors.")
        raise

# Models and Hyperparameters
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBClassifier': XGBClassifier(random_state=42, eval_metric='mlogloss'),
    'LGBMClassifier': LGBMClassifier(random_state=42, class_weight='balanced'),  # type: ignore
    'CatBoostClassifier': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')  # type: ignore
}

param_distributions: Dict[str, Dict[str, List[Union[int, float, str, None]]]] = {
    'RandomForestClassifier': {
        'select__k': [10, 15, 20, 25],
        'smote__k_neighbors': [1, 3, 5, 7],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 5]
    },
    'XGBClassifier': {
        'select__k': [10, 15, 20, 25],
        'smote__k_neighbors': [1, 3, 5, 7],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__gamma': [0, 0.1]
    },
    'LGBMClassifier': {
        'select__k': [10, 15, 20, 25],
        'smote__k_neighbors': [1, 3, 5, 7],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__num_leaves': [31, 62],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__reg_alpha': [0, 0.1],
        'classifier__reg_lambda': [0, 0.1]
    },
    'CatBoostClassifier': {
        'select__k': [10, 15, 20, 25],
        'smote__k_neighbors': [1, 3, 5, 7],
        'classifier__iterations': [100, 200, 300],
        'classifier__depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__l2_leaf_reg': [1, 3]
    }
}

def get_top_models(X: pd.DataFrame, y: Union[np.ndarray, Sequence[int]], n_top: int, phase: str = 'status', column_headers_json: Path = Path("src/column_headers.json")) -> List[Tuple[str, dict, float]]:

    # Helper Functions
    # Adjust categorical indices after feature selection
    def adjust_categorical_indices(selected_features: List[str], 
                                   original_categorical_indices: List[int], 
                                   original_columns: pd.Index) -> List[int]:
        """
        Map categorical feature indices after feature selection.

        Parameters
        ----------
        selected_features : list of str
            The feature names selected by SelectKBest.
        original_categorical_indices : list of int
            The categorical feature indices in the original dataset.
        original_columns : pd.Index
            All original column names.

        Returns
        -------
        list of int
            The indices of categorical features within the selected feature set,
            expressed relative to the selected features' order.
        """
        # Convert categorical indices -> categorical column names
        original_categorical_cols = [original_columns[i] for i in original_categorical_indices]

        # Now only keep those categorical cols that survived feature selection
        return [i for i, col in enumerate(selected_features) if col in original_categorical_cols]
    
    feature_cols, categorical_cols, _ = load_column_headers(column_headers_json, X)
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]
    original_categorical_cols = categorical_cols
    all_models = []
    total_iterations = sum(len(dist) for param_dist in param_distributions.values() for dist in param_dist.values())
    current_iteration = 0
    logger.info(f"Using {len(categorical_indices)} categorical indices: {categorical_indices}")

    # Group rare classes
#20250830    y_grouped, le = group_rare_classes(y, min_samples=round(0.02 * len(y)), phase=phase)
    y_grouped = np.array(y, dtype=int)
    cv = StratifiedKFold(n_splits=3 if phase == 'tier' else 3, shuffle=True, random_state=42)
    for model_name, param_dist in param_distributions.items():
        # Clone the model to get a fresh, unfitted instance
        model = clone(models[model_name])
        pipeline = ImbPipeline([
            ('select', SelectKBest(score_func=mutual_info_classif)),
            ('smote', SMOTENC(
                categorical_features=[], # will be set dynamically
                random_state=42,
                sampling_strategy='not majority'
                )),
                ('classifier', model)
        ])
        # Dynamically adjust categorical indices in param_dist
        new_params = {}
        for param_name, param_values in param_dist.items():
            if param_name == 'select__k':
                for raw_k in param_values:
                    if raw_k is None:
                        continue  # skip invalid
                    k = int(raw_k)
                    selected_features = X.columns[:k].tolist()
                    smote_features = adjust_categorical_indices(selected_features, categorical_indices, X.columns)
                    # assign list[int], not list[list[int]]
                    new_params['smote__categorical_features'] = cast(List[Union[int, float, str, None]], smote_features)
        param_dist.update(new_params)

        # Debug for XGB
        if model_name == 'XGBClassifier':
            logger.info(f"Debug XGB {phase}: Unique classes in y_grouped: {np.unique(y_grouped)}")
            logger.debug(f"BIN01: {phase = }\n{y_grouped = }")
            logger.info(f"Debug XGB {phase}: Class counts: {np.bincount(y_grouped)}")
        random_search = RandomizedSearchCV(
            pipeline,
            param_dist,
            cv=cv,
            n_iter = sum(len(v) for v in param_dist.values()),
            scoring='f1_macro',
            random_state=42,
            verbose=1,
            n_jobs=2,
            error_score='raise'
        )
        try:
            if model_name == 'XGBClassifier':
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_grouped)):
                    y_fold_train = y_grouped[train_idx]
                    y_fold_val = y_grouped[val_idx]
                    logger.info(f"Debug XGB {phase} Fold {fold}: Train classes {np.unique(y_fold_train)}, counts {np.bincount(y_fold_train)}")
                    logger.info(f"Debug XGB {phase} Fold {fold}: Val classes {np.unique(y_fold_val)}, counts {np.bincount(y_fold_val)}")
            random_search.fit(X, y_grouped)


            # Print the top features for the best model
            best_k = random_search.best_params_['select__k']
            selected_features = X.columns[:best_k].tolist()
            logger.info(f"\nTop {best_k} features for {model_name}:")
            logger.info(selected_features)

            for i in range(len(random_search.cv_results_['params'])):
                params = {k.replace('classifier__', ''): v for k, v in random_search.cv_results_['params'][i].items()}
                all_models.append((model_name, params, random_search.cv_results_['mean_test_score'][i]))
            current_iteration += len(random_search.cv_results_['params'])
            logger.info(f"Completed {current_iteration}/{total_iterations} iterations ({current_iteration/total_iterations*100:.2f}%)")
        except Exception as e:
            logger.critical(f"Error in {model_name} for {phase}: {str(e)}")
            raise ValueError(f"Error in {model_name} for {phase}: {str(e)}")

    all_models.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Top {n_top} models for {phase}:")
    for i, (model_name, params, score) in enumerate(all_models[:n_top]):
        logger.info(f"Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
    return all_models[:n_top]

# Main Execution
def main(args):
    try:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        logger.info(f"Loaded preprocessed data. Training size: {len(train_df)}, Test size: {len(test_df)}")
    except FileNotFoundError as e:
        logger.error(f"Error loading preprocessed data: {e}")
        raise

    feature_cols, categorical_cols, target_cols = load_column_headers(args.column_headers_json, train_df)

    # Filter available columns
    available_cols = [col for col in feature_cols if col in train_df.columns]
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in train_df: {missing_cols}")
    logger.info(f"Using available columns: {available_cols}")
    
    # Validate target columns
    status_target = 'final_contract_status_label'
    tier_target = 'final_contract_tier_label'
    if status_target not in target_cols or tier_target not in target_cols:
        logger.error(f"Target columns {status_target} or {tier_target} not found in column_headers.json Y=True columns")
        raise ValueError(f"Target columns {status_target} or {tier_target} not found in column_headers.json")
    
    # Phase 1: Status
    X_train_status = train_df[feature_cols]
    y_train_status = train_df[status_target].to_numpy()
    logger.debug(f"BIN00- {status_target = }\n{train_df[status_target] = }\n{y_train_status = }")
    for c in X_train_status.columns:
        if X_train_status[c].isnull().any():
            logger.debug(f"BIN01- {c = }, {X_train_status[c] = }")
    log_class_imbalance(y_train_status, "Status")
    
    top_models_status = get_top_models(X_train_status, y_train_status, 100, phase='status')
    for i, (model_name, params, score) in enumerate(top_models_status):
        logger.info(f"Status Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
        model = clone(models[model_name]) # Create a fresh clone of the model
        model.set_params(**params)
        model.fit(X_train_status, y_train_status)
        y_pred = model.predict(X_train_status)
        logger.info(f"Status Classification Report for {model_name}:")
        report: Dict[str, Dict[str, Any]] = cast(Dict[str, Dict[str, Any]], classification_report(y_train_status, y_pred, output_dict=True))    
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                logger.info(f"Class {class_name}:")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-score: {metrics['f1-score']:.4f}")
                if 'support' in metrics:
                    logger.info(f"  Support: {metrics['support']}")
        if 'macro avg' in report and isinstance(report['macro avg'], dict):
            logger.info("Macro Average Metrics:")
            logger.info(f"  Precision: {report['macro avg'].get('precision', 0):.4f}")
            logger.info(f"  Recall: {report['macro avg'].get('recall', 0):.4f}")
            logger.info(f"  F1-score: {report['macro avg'].get('f1-score', 0):.4f}")
        if 'weighted avg' in report and isinstance(report['weighted avg'], dict):
            logger.info("Weighted Average Metrics:")
            logger.info(f"  Precision: {report['weighted avg'].get('precision', 0):.4f}")
            logger.info(f"  Recall: {report['weighted avg'].get('recall', 0):.4f}")
            logger.info(f"  F1-score: {report['weighted avg'].get('f1-score', 0):.4f}")
        report_path = Path("logs")
        report_path.mkdir(parents=True, exist_ok=True)
        df_report = pd.DataFrame(report).transpose()
        report_file = report_path / "status_classification_report.csv"
        df_report.to_csv(report_file)
        logger.info(f"Saved Status classification report to {report_file}")

        log_feature_importance(model, available_cols)
        joblib.dump(model, f"models/status_model_{i+1}.pkl")
        logger.info(f"Status model {i+1} saved.")
    
    # Phase 2: Tier
    # Only consider approved contracts with valid tier labels. For now, 'rejected' is status_label = 0 but no tier assignment.
    train_approved_df = train_df[
        (train_df[status_target] == 0) & 
        (~train_df[tier_target].isin(["NA", "-1", "", "null", np.nan])) & 
        (train_df[tier_target].notnull())
    ]
    X_train_tier = train_approved_df[available_cols]
    y_train_tier = train_approved_df[tier_target].to_numpy()
    logger.debug(f"Tier00 - {status_target = }\n{train_approved_df[tier_target] = }\n{y_train_tier = }")
    log_class_imbalance(y_train_tier, "Tier")
    
    top_models_tier = get_top_models(X_train_tier, y_train_tier, 100, phase='tier')
    for i, (model_name, params, score) in enumerate(top_models_tier):
        logger.info(f"Tier Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
        model = clone(models[model_name]) # Create a fresh clone of the model
        model.set_params(**params)
        model.fit(X_train_tier, y_train_tier)
        y_pred = model.predict(X_train_tier)
        logger.info(f"Tier Classification Report for {model_name}:")
        report: Dict[str, Dict[str, Any]] = cast(Dict[str, Dict[str, Any]], classification_report(y_train_tier, y_pred, output_dict=True))    
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                logger.info(f"Class {class_name}:")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-score: {metrics['f1-score']:.4f}")
                if 'support' in metrics:
                    logger.info(f"  Support: {metrics['support']}")
        if 'macro avg' in report and isinstance(report['macro avg'], dict):
            logger.info("Macro Average Metrics:")
            logger.info(f"  Precision: {report['macro avg'].get('precision', 0):.4f}")
            logger.info(f"  Recall: {report['macro avg'].get('recall', 0):.4f}")
            logger.info(f"  F1-score: {report['macro avg'].get('f1-score', 0):.4f}")
        if 'weighted avg' in report and isinstance(report['weighted avg'], dict):
            logger.info("Weighted Average Metrics:")
            logger.info(f"  Precision: {report['weighted avg'].get('precision', 0):.4f}")
            logger.info(f"  Recall: {report['weighted avg'].get('recall', 0):.4f}")
            logger.info(f"  F1-score: {report['weighted avg'].get('f1-score', 0):.4f}")
        report_path = Path("logs")
        report_path.mkdir(parents=True, exist_ok=True)
        df_report = pd.DataFrame(report).transpose()
        report_file = report_path / "tier_classification_report.csv"
        df_report.to_csv(report_file)
        logger.info(f"Saved Tier classification report to {report_file}")
        log_feature_importance(model, available_cols)
        joblib.dump(model, f"models/tier_model_{i+1}.pkl")
        logger.info(f"Tier model {i+1} saved.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate various models for best fit to the data.")
    parser.add_argument("--train_csv", type=Path)
    parser.add_argument("--test_csv", type=Path)
    parser.add_argument("--column_headers_json", required=True, type=Path)
    args = parser.parse_args()

    Path("models/encoders").mkdir(parents=True, exist_ok=True)
    main(args)
    logger.info("Evaluation complete.")
    sys.exit(0)