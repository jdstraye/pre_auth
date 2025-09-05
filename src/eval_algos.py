"""
Evaluates machine learning models for status and tier classification using SMOTE,
feature selection, and StratifiedKFold. Saves top models and generates:
 - Worst-case CV fold report for the best model
 - Final test report for the best model
"""
import sys
import numpy as np
import pandas as pd
import logging
import joblib
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Union, Sequence, Any, Dict, cast, TypedDict
from sklearn import pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
#2020830 from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from utils import check_df_columns, sanitize_column_name, CustomRotatingFileHandler

# --- Global constants ---
RANDOM_STATE = 42

# Logging Setup
Path("logs").mkdir(exist_ok=True)
handler = CustomRotatingFileHandler("logs/pre_auth_eval_algos.log", maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.info("Logger initialized for eval_algos.py.")


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

def save_report(y_true, y_pred, prefix: str):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_file = Path("logs") / f"{prefix}_report.csv"
    df_report.to_csv(report_file)
    logger.info(f"Saved {prefix} classification report to {report_file}")
    return report_dict

def extract_valid_tier_records(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a full df of records into a df of valid tiers. To be a valid tier, the status must be approved or refunded
    and the final_contract_tier_label must exist. For example, "rejected" are approved, but they do not receive a
    tier, so they must be excluded from the tier_df.
    """

    # Setup
    tier_target = 'final_contract_tier_label'

    # Convert to string
    tier_str = in_df[tier_target].astype(str)

    # Filter out records that are not approved or that were approved but never got a tier (e.g., rejected)
    in_approved_df = in_df[
        (~tier_str.isin(["NA", "-1", "", "null"])) &
        (in_df[tier_target].notnull())
    ]
    return in_approved_df


# Models and Hyperparameters
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'XGBClassifier': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
    'LGBMClassifier': LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'CatBoostClassifier': CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced', thread_count=-1)
}

# Hyperparameter grids
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
        'classifier__l2_leaf_reg': [1, 3],
        'classifier__thread_count': [-1]
    }
}


# --- Core Model Search ---
def get_top_models(X: pd.DataFrame, y: Union[np.ndarray, Sequence[int]], n_top: int, phase: str = 'status', column_headers_json: Path = Path("src/column_headers.json")) -> Tuple[List[Tuple[str, dict[str, Any], float]], dict[str, Any], dict[str, Any]]:

    # Helper Functions
    # Adjust categorical indices after feature selection
    def adjust_categorical_indices(selected_features: List[str], 
                                   original_categorical_indices: List[int], 
                                   original_columns: pd.Index) -> List[int]:
        """
        Map categorical feature indices after feature selection.
        Returns indices relative to the *selected feature set*. 
        The naive approach returns indices relative to the initial feature set.

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
        smote_feature_indices = [
            i for i, col in enumerate(selected_features)
            if col in original_categorical_cols
        ]
        smote_feature_names = [selected_features[i] for i in smote_feature_indices]
        logger.debug(f"Adjusted SMOTE categorical feature names: {[selected_features[i] for i in smote_feature_indices]}")
        return smote_feature_indices

    # Compute total_iterations (distributive property)
    # For each model: #k_values * n_iter_per_k
    def n_iter_for_model(param_dist: Dict[str, List[Any]]) -> int:
        # other_dists length sum: we will use sum(len(v) for v in other_dists.values()) as the Randomized n_iter
        other_dists_len = sum(len(v) for p, v in param_dist.items() if p not in ('select__k', 'smote__categorical_features'))
        return max(1, other_dists_len)

    total_iterations = sum(
        len(param_dist.get('select__k', [X.shape[1]])) * n_iter_for_model(param_dist)
        for param_dist in param_distributions.values()
    )
    current_iteration = 0


    y_grouped = np.array(y, dtype=int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    all_models: List[Tuple[str, Dict[str, Any], float]] = []
    feature_cols, categorical_cols, _ = load_column_headers(column_headers_json, X)
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]



    # Group rare classes
#20250830    y_grouped, le = group_rare_classes(y, min_samples=round(0.02 * len(y)), phase=phase)
    # prepare data / cv
    y_grouped = np.array(y, dtype=int)
    cv = StratifiedKFold(n_splits=3 if phase == 'tier' else 3, shuffle=True, random_state=42)
    y_grouped = np.array(y, dtype=int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    all_models: List[Tuple[str, Dict[str, Any], float]] = []
    best_model_name =  None
    best_params = None
    best_score = -np.inf
    best_reports: List[Dict[str, Any]] = []
    logger.info(f"Using {len(categorical_indices)} categorical indices: {categorical_indices}")
    logger.info("Search strategy: Grid over select__k, Random over other hyperparameters (per-k).")

    for model_name, param_dist in param_distributions.items():
        model = clone(models[model_name])

        # Extract k values (exhaustive grid search) and other distributions (random sampling)
        k_values = [int(v) for v in param_dist.get('select__k', [X.shape[1]]) if v is not None]

        # other_dists: everything except select__k and smote__categorical_features
        other_dists_raw = {p: list(v) for p, v in param_dist.items() if p not in ('select__k', 'smote__categorical_features')}

        # Filter other_dists to only keys acceptable to this pipeline/classifier.
        # For classifier params ensure they are valid classifier params (without prefix).
        ## When done:
        ##  - classifer_paramnames: The parameters that the model accepts, so these should be randomized, but they don't have the prefix (e.g., classifier___)
        ##  - filtered_other_dists: Other parameters (not select___k or smote___categorical_features) that should be randomized (e.g, all "classifier___" params appropriate to the model and others, like "smote___k_neighbors")
        classifier_paramnames = set(k for k in model.get_params().keys())
        if model_name == "CatBoostClassifier":
            classifier_paramnames |= {
                "iterations",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "random_state",
                "verbose",
                "auto_class_weights",
                "thread_count",
                "bagging_temperature",
                "border_count",
                "random_strength",
                "min_data_in_leaf",
                "max_bin",
                "grow_policy",
                "one_hot_max_size",
                "leaf_estimation_iterations",
                "leaf_estimation_method",
            }
        filtered_other_dists: Dict[str, List[Any]] = {}
        for p, vals in other_dists_raw.items():
            if p.startswith('classifier__'):
                _, cls_param = p.split('__', 1)
                if cls_param in classifier_paramnames:
                    filtered_other_dists[p] = vals
                else:
                    logger.debug(f"Skipping {p} for model {model_name} (not a valid classifier param).")
            else:
                # keep non-classifier keys (e.g. 'smote__k_neighbors')
                filtered_other_dists[p] = vals

        logger.info(
            f"Model {model_name}: Grid size {len(k_values)=}, Random dims {len(filtered_other_dists)=} "
            f"(sizes: {[ (p, len(v)) for p, v in filtered_other_dists.items() ]})"
        )

        # for each k (grid)
        for raw_k in k_values:
            try:
                k = int(raw_k)  # ensure int; avoids Pylance type complains
            except Exception:
                logger.warning(f"Skipping invalid select__k value: {raw_k}")
                continue

            # build pipeline for this k
            candidate_pipeline = ImbPipeline([
                ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
                ('smote', SMOTENC(categorical_features=[],
                    random_state=RANDOM_STATE,
                    sampling_strategy='not majority'
                    )),
                ('classifier', model)
            ])

            # Fit SelectKBest on full X (we do selection once per k on full data)
            logging.debug(f"{k = }: {candidate_pipeline = }")
            candidate_pipeline.named_steps['select'].fit(X, y_grouped)
            logging.debug(f"{k = }: {candidate_pipeline.named_steps['select'] = }")
            selected_features_mask = candidate_pipeline.named_steps['select'].get_support()
            logging.debug(f"{k = }: {selected_features_mask = }")
            selected_feature_indices = np.where(selected_features_mask)[0]  # Get indices of selected features
            selected_features = X.columns[selected_feature_indices].tolist()

            # Adjust categorical indices
            ## Map original categorical indices -> indices in the selected feature set
            logger.debug(f"FI00a: {selected_feature_indices = }")
            # Determine SMOTE categorical positions relative to selected_features (0..k-1)
            smote_feature_indices = adjust_categorical_indices(
                selected_features,
                [int(i) for i in categorical_indices if isinstance(i, int)],
                X.columns
            )
            logger.debug(f"FI00b: {smote_feature_indices = }")

            # Guard: only keep indices within [0, k-1]
            smote_feature_indices = [int(i) for i in smote_feature_indices if 0 <= int(i) < int(k)]

            logger.debug(f"FI01a: {k = }, selected_feature_count={len(selected_features)}")
            logger.debug(f"FI01b: smote_feature_indices ({k = }) = {smote_feature_indices}")
            logger.info(f"Model {model_name} {k = }: {len(selected_features) = }\n {len(smote_feature_indices) = } {smote_feature_indices = }")

            # set SMOTE categorical indices as fixed pipeline params (do not include in RandomizedSearch candidates - do NOT put in param_distributions)
            candidate_pipeline.set_params(smote__categorical_features=smote_feature_indices)

            # n_iter for RandomizedSearch on 'other' dims
            n_iter_rs = n_iter_for_model(param_dist)

            # Build RandomizedSearchCV only with filtered_other_dists
            random_search = RandomizedSearchCV(
                estimator=candidate_pipeline,
                param_distributions=filtered_other_dists,
                cv=cv,
                n_iter = n_iter_rs,
                scoring='f1_macro',
                random_state=RANDOM_STATE,
                verbose=1,
                n_jobs=2,
                error_score='raise'
            )

            # Fit randomized search for this (model_name, k)
            try:
                random_search.fit(X, y_grouped)

                # Record results:
                # - inject the fixed params so they appear in logs
                # - keep pipeline-style keys with a prefix, e.g., classifier__..., select__k, smote__categorical_features
                # increment iteration counter by number of candidates RandomizedSearch actually tried
                tried = len(random_search.cv_results_['params'])
                for i in range(tried):
                    candidate_params = dict(random_search.cv_results_['params'][i])
                    # inject bookkeeping params (keep prefix for classifier). Keep keys as pipeline-style (no stripping)
                    candidate_params['select__k'] = k
                    candidate_params['smote__categorical_features'] = smote_feature_indices
                    score = random_search.cv_results_['mean_test_score'][i]
                    all_models.append((model_name, candidate_params, score))

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_params = candidate_params
                    # collect fold reports for this candidate
                    best_reports = []
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_grouped)):
                        y_val = y_grouped[val_idx]
                        X_val = X.iloc[val_idx]
                        fold_pipeline = clone(candidate_pipeline)
                        fold_pipeline.set_params(**candidate_params)
                        fold_pipeline.fit(X.iloc[train_idx], y_grouped[train_idx])
                        y_pred_fold = fold_pipeline.predict(X_val)
                        best_reports.append(cast(Dict[str, Any], classification_report(y_val, y_pred_fold, output_dict=True)))
                # update progress counters and log succinctly
                current_iteration += tried
                pct = (current_iteration / total_iterations * 100) if total_iterations else 100.0
                logger.info(f"{phase}-Completed {current_iteration}/{total_iterations} iterations ({pct:.2f}%) [{model_name = }, {k = },{tried = }]")

            except Exception as e:
                logger.critical(f"Error in {model_name} for {phase = } with {k = }: {e}")
                raise ValueError(f"Error in {model_name} for {phase}: {str(e)}")

        # after all k for this model, debug final state
        logger.debug(
            f"Final SMOTE categorical_features for {model_name}: "
            f"{candidate_pipeline.get_params().get('smote__categorical_features') = }\n"
            f"{candidate_pipeline.get_params().get('select__k') = }"
        )
        logger.debug(f"Final filtered/random dists for {model_name}: {filtered_other_dists}\n")

    # Sort
    all_models.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Top {n_top} models for {phase}:")
    for i, (model_name, params, score) in enumerate(all_models[:n_top]):
        logger.info(f"Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")

    # worst-case fold (lowest macro f1) of the best overall scoring model
    worst_case_report = min(best_reports, key=lambda r: r['macro avg']['f1-score']) if best_reports else {}

    return all_models[:n_top], {"model": best_model_name, "params": best_params, "score": best_score}, worst_case_report

# --- Main Execution ---
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
    top_models_status, best_status, worst_case_report = get_top_models(X_train_status, y_train_status, 100, "status", column_headers_json=args.column_headers_json)

    # Report all the top models
    for i, (model_name, params, score) in enumerate(top_models_status):
        logger.info(f"Status Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
        pipeline = ImbPipeline([
            ('select', SelectKBest(score_func=mutual_info_classif, k=params['select__k'])),
            ('smote', SMOTENC(
                categorical_features=params['smote__categorical_features'],
                random_state=42,
                sampling_strategy='not majority'
            )),
            ('classifier', clone(models[model_name]))
        ])
        pipeline.set_params(**params)
        pipeline.fit(X_train_status, y_train_status)
        y_pred = pipeline.predict(X_train_status)
        logger.info(f"Status Classification Report for {model_name}:")
        report: Dict[str, Dict[str, Any]] = cast(Dict[str, Dict[str, Any]], classification_report(y_train_status, y_pred, output_dict=True))    
        for class_name, metrics in report.items():
            logger.info(f"***** {model_name = } {params = } *****")
            if isinstance(metrics, dict) and 'precision' in metrics:
            
                logger.info(f"  Class {class_name}:")
                logger.info(f"    Precision: {metrics['precision']:.4f}")
                logger.info(f"    Recall: {metrics['recall']:.4f}")
                logger.info(f"    F1-score: {metrics['f1-score']:.4f}")
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

        log_feature_importance(pipeline.named_steps['classifier'], available_cols)

    # Build the pipeline with the best k-fold score parameters
    best_status_pipeline = ImbPipeline([
        ('select', SelectKBest(score_func=mutual_info_classif, k=best_status['params']['select__k'])),
        ('smote', SMOTENC(categorical_features=best_status['params']['smote__categorical_features'], random_state=RANDOM_STATE, sampling_strategy='not majority')),
        ('classifier', clone(models[best_status['model']]))
    ])
    best_status_pipeline.set_params(**best_status['params'])
    best_status_pipeline.fit(X_train_status, y_train_status)
    y_test_pred = best_status_pipeline.predict(test_df[feature_cols])
    save_report(test_df[status_target], y_test_pred, "status_test")
    joblib.dump(best_status_pipeline, "models/status_best.pkl")
    logger.info(f"Best Status Model: {best_status}")
    pd.DataFrame(worst_case_report).transpose().to_csv("logs/status_cv_worstcase_report.csv")

    # --- Phase 2: Tier ---
    # Only consider approved contracts with valid tier labels.
    train_approved_df = extract_valid_tier_records(train_df)
    X_train_tier = train_approved_df[feature_cols]
    y_train_tier = train_approved_df[tier_target].astype(int).to_numpy()
    logger.debug(f"Tier00 - {status_target = }\n{train_approved_df[tier_target] = }\n{y_train_tier = }")
    log_class_imbalance(y_train_tier, "Tier")
    top_models_tier, best_tier, worst_tier_report = get_top_models(X_train_tier, y_train_tier, 100, "tier", args.column_headers_json)
    logger.info(f"Best Tier Model: {best_tier}")
    pd.DataFrame(worst_tier_report).transpose().to_csv("logs/tier_cv_worstcase_report.csv")
    for i, (model_name, params, score) in enumerate(top_models_tier):
        logger.info(f"Tier Model {i+1}: {model_name}, Params: {params}, Score: {score:.4f}")
        clean_params = params.copy()
        clean_k = clean_params.pop('select__k', None)
        smote_cats = clean_params.pop('smote__categorical_features', None)
        pipeline = ImbPipeline([
            ('select', SelectKBest(score_func=mutual_info_classif, k=clean_k)),
            ('smote', SMOTENC(
                categorical_features=smote_cats,
                random_state=42,
                sampling_strategy='not majority'
            )),
            ('classifier', clone(models[model_name]))
        ])
        pipeline.set_params(**clean_params)
        pipeline.fit(X_train_tier, y_train_tier)
        y_pred = pipeline.predict(X_train_tier)
        logger.info(f"Tier Classification Report for {model_name} / {params = }:")
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
        log_feature_importance(pipeline.named_steps['classifier'], available_cols)

    best_tier_pipeline = ImbPipeline([
        ('select', SelectKBest(score_func=mutual_info_classif, k=best_tier['params']['select__k'])),
        ('smote', SMOTENC(categorical_features=best_tier['params']['smote__categorical_features'], random_state=RANDOM_STATE, sampling_strategy='not majority')),
        ('classifier', clone(models[best_tier['model']]))
    ])
    best_tier_pipeline.set_params(**best_tier['params'])
    best_tier_pipeline.fit(X_train_tier, y_train_tier)
    test_tier_df = extract_valid_tier_records(test_df)
    y_test_pred_tier = best_tier_pipeline.predict(test_tier_df[feature_cols])
    save_report(test_tier_df[tier_target], y_test_pred_tier, "tier_test")
    joblib.dump(best_tier_pipeline, "models/tier_best.pkl")
    pd.DataFrame(worst_tier_report).transpose().to_csv("logs/tier_cv_worstcase_report.csv")
    logger.info(f"Best Tier Model: {best_tier}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate various models for best fit to the data.")
    parser.add_argument("--train_csv", type=Path)
    parser.add_argument("--test_csv", type=Path)
    parser.add_argument("--column_headers_json", required=True, type=Path)
    args = parser.parse_args()

    main(args)
    logger.info("Evaluation complete.")
    sys.exit(0)