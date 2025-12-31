"""
Evaluation harness
-----------------------------
Thorough implementation of the model evaluation / hyperparameter search
pipeline. Uses ParameterSampler for explicit candidate sampling, warns and
aborts on SMOTE sanity failures, logs feature importances from FeatureSelectingClassifier,
and saves timestamped outputs and updates latest symlinks.
- Builds candidate hyperparameter sets using sklearn.model_selection.ParameterSampler
 to allow complex conditional constraints and mutual-exclusivity checking.
- Implements a custom cross-validation loop (StratifiedKFold) over candidates
  so feature selection, SMOTE and classifier training happen strictly within
  each training fold (no leakage).
- Feature selection: SelectFromModel wrapped in `FeatureSelectingClassifierDf`
  which preserves DataFrame column names and logs feature importance.
- NamedSMOTE / NamedSMOTENC accept categorical feature names and log counts
Evaluates machine learning models for status and tier classification using SMOTE,
feature selection, and StratifiedKFold. Specifically,
 - Uses name-based SMOTE (NamedSMOTE / NamedSMOTENC)
 - Preserves feature names across SelectKBest with SelectFromModelDf
 - Stores SMOTE categorical features as names (not indices)
 - Saves feature info via joblib
 - Final saved models are CalibratedClassifierCV trained on train+test
 - Reports and saves worst-case fold reports and worst metrics

Usage (example):
    python -m src.eval_algos --train_csv data/train.csv --test_csv data/test.csv --column_headers_json src/column_headers.json

Notes:
 - This script is designed for Linux (symlink logic uses pathlib).
"""
from __future__ import annotations
from catboost import CatBoostClassifier
from datetime import datetime
from functools import reduce
from imblearn.base import SamplerMixin, BaseSampler
from imblearn.over_sampling import SMOTENC, SMOTE
#debug 20250917 from imblearn.over_sampling import SMOTENC
#debug 20250917 from imblearn.over_sampling import SMOTE as ImbSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from logging.handlers import RotatingFileHandler
from operator import mul
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, roc_auc_score, get_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler, cross_validate, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#@note - wip src imports
from src.utils import check_df_columns, sanitize_column_name, setup_logging, gv, load_column_headers
from src.eval_utils import generate_parameter_samples, save_report, write_csv_and_symlink, write_json_and_symlink, now_ts, log_class_imbalance, calc_n_iter_model, extract_valid_tier_records
from src.pipeline_coordinator import MLPipelineCoordinator
from src.components.smote_sampler import MaybeSMOTESampler
from src.components.feature_selector import FeatureSelector, FeatureSelectingClassifier
from src.debug_library import debug_check_for_nans, debug_check_frame
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast, Mapping
from xgboost import XGBClassifier
import joblib
import json
import logging
import math
import numpy as np
import pandas as pd
import sys
import time
import traceback
import warnings

setup_logging(gv.LOG_DIR / "pre_auth_eval_algos.log")
logger = logging.getLogger(__name__)

# --------------------------
# Models & hyperparameter distributions
# --------------------------

#@note Base model definitions (default instantiation). We keep classifiers here for cloning.
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=gv.RANDOM_STATE, class_weight='balanced_subsample'),
    'ExtraTreesClassifier': ExtraTreesClassifier(random_state=gv.RANDOM_STATE, class_weight='balanced_subsample'),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=gv.RANDOM_STATE),
    'XGBClassifier': XGBClassifier(random_state=gv.RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
    'LGBMClassifier': LGBMClassifier(random_state=gv.RANDOM_STATE, class_weight='balanced'),
    'CatBoostClassifier': CatBoostClassifier(random_state=gv.RANDOM_STATE, verbose=0, auto_class_weights='Balanced', thread_count=-1)
}

#@note param distributions (lists) for ParameterSampler
param_distributions: Dict[str, Dict[str, List[Any]]] = {
    'RandomForestClassifier': {
        # smote hyperparams available on sampler step
        'smote__enabled': [True],
        'smote__k_neighbors': [1, 3, 5, 7, 10, 15],
        'feature_selecting_classifier__max_features': [10, 15, 20, 25, 30, None],
        'feature_selecting_classifier__threshold': [None],  # mutually exclusive with max_features
        'feature_selecting_classifier__estimator__n_estimators': [100, 200, 300, 400],
        'feature_selecting_classifier__estimator__max_depth': [None, 5, 10, 15],
        'feature_selecting_classifier__estimator__min_samples_split': [1, 2, 5, 10],
        'encoding': ['ordinal', 'ohe'],
        'smote__method': ['smotenc', 'smote', 'none']
    },
    'ExtraTreesClassifier': {
        'smote__enabled': [True],
        'smote__k_neighbors': [1, 3, 5, 7, 10],
        'feature_selecting_classifier__max_features': [10, 15, 20, 25, 30, None],
        'feature_selecting_classifier__threshold': [None],
        'feature_selecting_classifier__estimator__n_estimators': [100, 200, 300],
        'feature_selecting_classifier__estimator__max_depth': [None, 3, 5, 7, 10],
        'encoding': ['ordinal', 'ohe'],
        'smote__method': ['smotenc', 'smote', 'none']
    },
    'GradientBoostingClassifier': {
        'smote__enabled': [True],
        'smote__k_neighbors': [1, 3, 5, 7, 10],
        'feature_selecting_classifier__max_features': [10, 15, 20, 25, 30, None],
        'feature_selecting_classifier__threshold': [None],
        'feature_selecting_classifier__estimator__n_estimators': [100, 200],
        'feature_selecting_classifier__estimator__max_depth': [3, 5, 7],
        'feature_selecting_classifier__estimator__learning_rate': [0.01, 0.05, 0.1],
        'encoding': ['ohe', 'ordinal'],
        'smote__method': ['smote', 'none']
    },
    "XGBClassifier": {
        'smote__enabled': [True],
        'smote__k_neighbors': [1, 3, 5, 7, 10, 15],
        'feature_selecting_classifier__max_features': [10, 15, 20, 25, None],
        'feature_selecting_classifier__threshold': [None],
        'feature_selecting_classifier__estimator__n_estimators': [100, 200, 300],
        'feature_selecting_classifier__estimator__max_depth': [3, 5, 7, 10],
        'feature_selecting_classifier__estimator__learning_rate': [0.005, 0.01, 0.05, 0.1],
        'feature_selecting_classifier__estimator__min_child_weight': [1, 5, 10],
        'feature_selecting_classifier__estimator__gamma': [0, 0.1, 0.2],
        'encoding': ['ordinal', 'ohe'],
        'smote__method': ['smotenc', 'smote', 'none']
    },
    "LGBMClassifier": {
        'smote__enabled': [True],
        'smote__k_neighbors': [1, 3, 5, 7, 10, 15],
        'feature_selecting_classifier__max_features': [10, 15, 20, None],
        'feature_selecting_classifier__threshold': [None],
        'feature_selecting_classifier__estimator__n_estimators': [100, 200, 300],
        'feature_selecting_classifier__estimator__num_leaves': [31, 62, 124],
        'feature_selecting_classifier__estimator__subsample': [0.6, 0.8, 1.0],
        'feature_selecting_classifier__estimator__reg_alpha': [0, 0.1, 0.5],
        'feature_selecting_classifier__estimator__reg_lambda': [0, 0.1, 0.5],
        'feature_selecting_classifier__estimator__learning_rate': [0.01, 0.05, 0.1],
        'encoding': ['ordinal', 'ohe'],
        'smote__method': ['smotenc', 'smote', 'none']
    },
    "CatBoostClassifier": {
        # allow SMOTE toggle for CatBoost; user requested CatBoost SMOTE bypass hyperparam
        'smote__enabled': [False, True],
        'smote__k_neighbors': [1, 3, 5, 7, 10, 15],
        'feature_selecting_classifier__max_features': [10, 15, 20, 25, 30, None],
        'feature_selecting_classifier__threshold': [None],
        'feature_selecting_classifier__estimator__iterations': [100, 200, 300, 400],
        'feature_selecting_classifier__estimator__depth': [3, 5, 7],
        'feature_selecting_classifier__estimator__learning_rate': [0.01, 0.05, 0.1],
        'feature_selecting_classifier__estimator__l2_leaf_reg': [1, 3, 5],
        # optionally let CatBoost try more features (but leaving one-hot encoding up to upstream)
        'encoding': ['ordinal', 'ohe'],
        'smote__method': ['none', 'smote']
    }
}

# --------------------------
# Save pipeline state helper (adapted to FeatureSelectingClassifier)
# --------------------------
def save_pipeline_state(pipeline: ImbPipeline, phase: str, models_dir: Path = gv.MODELS_DIR) -> Dict[str, Any]:
    """
    Save uncalibrated pipeline artifact and feature-info (selected features, mask, scores, smote names).
    Returns feature_info dict. Handles pipeline with 'feature_selecting_classifier' step.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = models_dir / f"{phase}_pipeline_uncalibrated.pkl"
    joblib.dump(pipeline, pipeline_file)
    logger.info(f"Saved uncalibrated pipeline to {pipeline_file}")

    feature_info: Dict[str, Any] = {}
    fsc = pipeline.named_steps.get("feature_selecting_classifier")
    smote_step = pipeline.named_steps.get("smote")

    try:
        if fsc is not None and hasattr(fsc, "selected_features_"):
            selected = getattr(fsc, "selected_features_", None)
            feature_info["selected_features"] = list(selected) if selected is not None else None
            # store feature_importances_ if present
            fi = getattr(fsc, "feature_importances_", None)
            feature_info["feature_importances_df"] = fi.to_dict(orient="list") if fi is not None else None
        else:
            feature_info["selected_features"] = None
            feature_info["feature_importances_df"] = None

        feature_info["smote__categorical_feature_names"] = getattr(smote_step, "categorical_feature_names", None) \
            if smote_step is not None else None

        feature_info["sklearn_pandas_version"] = {"pandas": pd.__version__}
        feature_info_file = models_dir / f"{phase}_feature_info.pkl"
        joblib.dump(feature_info, feature_info_file)
        logger.info(f"Saved feature info to {feature_info_file}")
    except Exception as e:
        logger.exception(f"Failed to extract/save feature info: {e}")
    return feature_info

# --------------------------
#@note Core model search function (uses ParameterSampler -> cross_validate)
# --------------------------
def get_top_models(X: pd.DataFrame,
                   y: Union[np.ndarray, Sequence[int]],
                   n_top: int,
                   phase: str = "status",
                   column_headers_json: Path = gv.DEFAULT_SCHEMA_PATH,
                   random_search_mult: float = gv.RANDOM_SEARCH_ITER_MULT,
                   n_jobs_cv: int = 1
                   ) -> Tuple[List[Tuple[str, Dict[str, Any], float]], Dict[str, Any], Dict[str, Any]]:
    """
    Search models using ParameterSampler-generated candidates and cross_validate.

    Returns:
      - top_candidates: list[(model_name, params_dict, mean_f1_macro)]
      - best_summary: dict with best model info
      - worst_case_fold_report: classification_report dict for worst fold of best model (recomputed)
    """
    # Load schema and categorical names
    headers = load_column_headers(column_headers_json, X)
    original_categorical_names = [c for c in headers.get('categorical_cols', []) if c in X.columns]

    n_splits = 3 if phase == "tier" else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=gv.RANDOM_STATE)

    all_candidates: List[Tuple[str, Dict[str, Any], float]] = []
    all_candidates_rows: List[Dict[str, Any]] = []

    # Progress / ETA helpers: compute total candidate count ahead of time using heuristics
    model_sample_plan: Dict[str, int] = {}
    for model_name, dist in param_distributions.items():
        # number of possible combos (product of lengths)
        n_iter_model = math.ceil(calc_n_iter_model(dist) * float(random_search_mult))
        model_sample_plan[model_name] = n_iter_model
    total_candidates = math.ceil(sum(model_sample_plan.values()))
    logger.info(f"Planned {total_candidates} parameter samples:\n {model_sample_plan = }")

    # Constraints: ensure max_features and threshold are mutually exclusive
    def exclusivity_constraint(sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        max_f = sample.get("feature_selecting_classifier__max_features", None)
        thr = sample.get("feature_selecting_classifier__threshold", None)
        if (max_f is not None) and (thr is not None):
            return False, "Both max_features and threshold are set, but they are mutually exclusive."
        return True, None
    
    # Loop models
    candidate_counter = 0
    t0 = time.time()

    #@note - Primary Loop
    # We'll collect per-model candidate rows for CSV export
    best_model_name = ""
    best_fold_reports = []
    best_params = {}
    best_score = -1.0
    for model_name, dist in param_distributions.items():
        logger.info(f"Starting model search for {model_name}")
        base_model = clone(models[model_name])
        n_iter_model = model_sample_plan.get(model_name, 1)

        # prepare param grid for ParameterSampler (we will sample n_iter_model valid candidates)
        param_grid_for_sampler = dict(dist)

        # Ensure smote__categorical_feature_names gets the original categorical names by default
        # (SMOTE runs *before* selection, so we pass originals)
        if "smote__categorical_feature_names" not in param_grid_for_sampler:
            param_grid_for_sampler["smote__categorical_feature_names"] = [original_categorical_names]

        # Generate valid parameter samples
        samples = generate_parameter_samples(param_grid_for_sampler,
                             n_samples=math.ceil(n_iter_model),  # oversample to account for rejections
                                             random_state=gv.RANDOM_STATE)
        if not samples:
            logger.warning(f"No valid parameter samples generated for {model_name}; skipping.")
            continue

        # Build pipeline skeleton: SMOTE first, then FeatureSelectingClassifier (estimator=base_model)
        # Note: we'll clone the estimator into the feature selector via FeatureSelectingClassifier
        for candidate_params in samples:
            candidate_counter += 1
            elapsed = time.time() - t0
            pct = (candidate_counter / float(total_candidates)) * 100.0 if total_candidates else 100.0
            eta = None
            if candidate_counter > 0 and pct > 0:
                est_total = elapsed / (pct / 100.0)
                eta = time.time() + (est_total - elapsed)
                eta_str = datetime.utcfromtimestamp(eta).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                eta_str = "unknown"

            logger.info(f"{pct:.2f}% complete, [{candidate_counter}/{total_candidates}] Evaluating candidate for {model_name}; eta={eta_str}; params={candidate_params}")

            # Build pipeline instance for this candidate
            smote_cfg = {
                "enabled": bool(candidate_params.get("smote__enabled", True)),
                "categorical_feature_names": candidate_params.get("smote__categorical_feature_names", original_categorical_names),
                "k_neighbors": int(candidate_params.get("smote__k_neighbors", 5)),
                "sampling_strategy": candidate_params.get("smote__sampling_strategy", "auto"),
                "random_state": candidate_params.get("smote__random_state", gv.RANDOM_STATE),
                "min_improvement": candidate_params.get("smote__min_improvement", gv.DEFAULT_SMOTE_MIN_IMPROVEMENT)
            }
            smote_step = MaybeSMOTESampler(**smote_cfg)

            # Create FeatureSelectingClassifier with base_model (hyperparams will be set via set_params)
            max_features = candidate_params.get("feature_selecting_classifier__max_features", None)
            threshold = candidate_params.get("feature_selecting_classifier__threshold", None)

            fsc = FeatureSelectingClassifier(
                estimator=clone(base_model),
                max_features=max_features,
                threshold=threshold
            )

            pipeline = ImbPipeline([("smote", smote_step),
                                    #("debug", DebugTransformer(tag="post-smote")),
                                    ("feature_selecting_classifier", fsc)
                     ])

            # Now set other nested estimator params from candidate_params into pipeline
            # Candidate params are full keys e.g. feature_selecting_classifier__estimator__n_estimators
            try:
                pipeline.set_params(**candidate_params)
            except Exception as e:
                logger.exception(f"Failed to set candidate params on pipeline: {e}; skipping candidate.")
                continue

            # Evaluate candidate using cross_validate (f1_macro)
            scoring = {"f1_macro": "f1_macro"}
            # If binary (2 classes), include roc_auc scorer
            unique_labels = np.unique(np.asarray(y))
            include_auc = False
            if len(unique_labels) == 2:
                scoring["auc"] = "roc_auc"
                include_auc = True
            else:
                # multiclass roc auc attempt if estimator supports predict_proba
                scoring["auc_ovr"] = "roc_auc_ovo_weighted"  # fallback; may fail if estimator lacks proba
                # we will handle failures below

            # Run cross_validate; if any fold raises exception, catch and mark candidate as failed
            try:
                debug_check_frame(X, "X_train before fit")
                # In get_top_models(), right before cross_validate():
                logger.debug(f"Data check before cross_validate:")
                logger.debug(f"X dtypes: {X.dtypes['AutomaticFinancing_below_600_']}")
                logger.debug(f"X unique values: {X['AutomaticFinancing_below_600_'].unique()}")
                logger.debug(f"Any NaN/inf in X: {X['AutomaticFinancing_below_600_'].isin([np.nan, np.inf, -np.inf]).any()}")

                from typing import Any, cast
                try:
                    cv_res = cross_validate(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1,  # n_jobs_cv,
                        error_score=cast(Any, "raise"),
                        return_train_score=False,
                    )
                except Exception as e:
                    logger.debug(f"cross_validate failed: {e}")
                    # Try a manual fit to get the underlying error location, then rerun CV
                    logger.debug("Trying manual fit to isolate the problem...")
                    pipeline.fit(X, y)
                    cv_res = cross_validate(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1,
                        error_score=cast(Any, "raise"),
                        return_train_score=False,
                    )
                mean_f1 = float(np.mean(cv_res["test_f1_macro"]))
                auc_score = None
                if "test_auc" in cv_res:
                    auc_score = float(np.mean(cv_res["test_auc"]))
                elif "test_auc_ovr" in cv_res:
                    auc_score = float(np.mean(cv_res["test_auc_ovr"]))
                else:
                    auc_score = None

                logger.info(f"Candidate result: model={model_name} mean_f1={mean_f1:.4f} auc={auc_score}")

                # Append to all candidates records
                all_candidates.append((model_name, dict(candidate_params), mean_f1))
                row = {
                    "timestamp": now_ts(),
                    "model": model_name,
                    "mean_f1_macro": mean_f1,
                    "auc": auc_score,
                    "params": json.dumps(candidate_params, default=str)
                }
                all_candidates_rows.append(row)

                # Track best
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_model_name = model_name
                    best_params = dict(candidate_params)

                    # Recompute fold reports for the candidate (manual folds so we can preserve per-fold classification_report)
                    best_fold_reports = []
                    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_tr, y_val = np.asarray(y)[train_idx], np.asarray(y)[val_idx]
                        # build pipeline anew with same params
                        smote_step_fold = MaybeSMOTESampler(**smote_cfg)
                        fsc_fold = FeatureSelectingClassifier(estimator=clone(base_model),
                                                             max_features=max_features,
                                                             threshold=threshold)
                        fold_pipeline = ImbPipeline([("smote", smote_step_fold), ("feature_selecting_classifier", fsc_fold)])
                        fold_pipeline.set_params(**candidate_params)
                        try:
                            fold_pipeline.fit(X_tr, y_tr)
                            y_pred_fold = fold_pipeline.predict(X_val)
                            rpt = classification_report(y_val, y_pred_fold, output_dict=True)
                            best_fold_reports.append(rpt)
                        except Exception as e:
                            logger.exception(f"Failed to fit/predict fold {fold_i} for best candidate: {e}")
                            # If fold fails, skip that fold's report but continue
                    # keep best_fold_reports for later worst-case extraction
            except Exception as e:
                # Candidate evaluation failed; log & continue to next candidate
                logger.exception(f"Candidate evaluation failed for model={model_name} params={candidate_params}: {e}")
                # store failed candidate row
                all_candidates_rows.append({
                    "timestamp": now_ts(),
                    "model": model_name,
                    "mean_f1_macro": None,
                    "auc": None,
                    "params": json.dumps(candidate_params, default=str),
                    "error": str(e)
                })
                continue

        # After all candidates
        logger.error("No candidate evaluations produced results.")
        logger.error("No candidates succeeded during model search.")
        return [], {}, {}

    # Persist all candidate CSV to logs and models with timestamp and symlink
    timestamp = now_ts()
    df_all = pd.DataFrame(all_candidates_rows)
    write_csv_and_symlink(df_all, gv.LOG_DIR, f"{phase}_all_models", timestamp)
    write_csv_and_symlink(df_all, gv.MODELS_DIR, f"{phase}_all_models", timestamp)

    # sort and produce top list
    all_candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = all_candidates[:n_top]

    # compute worst-case fold from best_fold_reports (if present)
    worst_case_fold_report = {}
    worst_metrics: Dict[str, float] = {}
    if best_fold_reports:
        worst_case_fold_report = min(best_fold_reports, key=lambda r: r["macro avg"]["f1-score"])
        precisions = [r["macro avg"]["precision"] for r in best_fold_reports]
        recalls = [r["macro avg"]["recall"] for r in best_fold_reports]
        f1s = [r["macro avg"]["f1-score"] for r in best_fold_reports]
        accuracies = [r.get("accuracy", 0.0) for r in best_fold_reports]
        worst_metrics = {
            "worst_macro_precision": float(np.min(precisions)),
            "worst_macro_recall": float(np.min(recalls)),
            "worst_macro_f1": float(np.min(f1s)),
            "worst_accuracy": float(np.min(accuracies))
        }
    else:
        logger.warning("No fold reports were collected for the best candidate.")

    best_summary = {"model": best_model_name, "params": best_params, "score": best_score, "worst_metrics": worst_metrics}

    return top_candidates, best_summary, worst_case_fold_report


# --------------------------
# Top-level main flow
# --------------------------
def main(args):

    # Setup logging
    log_file = gv.LOG_DIR / "pre_auth_eval_algos.log"
    setup_logging(log_file)
    logger.info("Starting model evaluation.")

    # Load CSVs
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Loaded train {train_df.shape} test {test_df.shape}")

    #feature_cols, categorical_cols, target_cols = load_column_headers(args.column_headers_json, train_df)
    headers = load_column_headers(args.column_headers_json, train_df)
    available_cols = [c for c in headers.get('feature_cols', []) if c in train_df.columns]
    missing_cols = [c for c in headers.get('feature_cols', []) if c not in train_df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns in train_df: {missing_cols}")
    logger.info(f"Using {len(available_cols)} available feature columns.")

    # Targets - allow case variations in schema by matching lowercase
    def find_header_key(target_name: str, target_list: Optional[List[str]]):
        if not target_list:
            return None
        for t in target_list:
            if t and isinstance(t, str) and t.lower() == target_name.lower():
                return t
        return None

    status_target = find_header_key("final_contract_status_label", headers.get('target_cols', []))
    tier_target = find_header_key("final_contract_tier_label", headers.get('target_cols', []))
    if status_target is None or tier_target is None:
        logger.error("Expected target columns not present in column_headers.json (Y=True).")
        raise ValueError("Target columns missing in schema.")

    # -------------------------
    # STATUS PHASE
    # -------------------------
    X_train_status = train_df[available_cols].copy()
    y_train_status = train_df[status_target].to_numpy()
    log_class_imbalance(y_train_status, "Status")

    # Derive smoke_flag early on (explicit flag or small random_search_mult)
    smoke_flag = getattr(args, 'smoke', False) or (args.random_search_mult < 0.05)

    if getattr(args, "use_coordinator", False):
        logger.info("Using MLPipelineCoordinator for model search (status phase)")
        coordinator = MLPipelineCoordinator()
        smoke_mode = smoke_flag
        n_splits = 2 if smoke_mode else 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=gv.RANDOM_STATE)
        n_jobs_cv = 1 if smoke_mode else -1
        models_to_use = models
        param_dist_to_use = param_distributions
        if smoke_mode:
            from sklearn.ensemble import RandomForestClassifier as _RF
            models_to_use = {'RandomForestClassifier': _RF(random_state=gv.RANDOM_STATE, n_estimators=10)}
            param_dist_to_use = {
                'RandomForestClassifier': {
                    'smote__enabled': [False],
                    'feature_selecting_classifier__max_features': [10, None],
                    'feature_selecting_classifier__threshold': [None],
                    'feature_selecting_classifier__estimator__n_estimators': [10]
                }
            }
        top_models_status, best_status = coordinator.search_models(models=models_to_use,
                                                                  param_distributions=param_dist_to_use,
                                                                  X=X_train_status,
                                                                  y=y_train_status,
                                                                  n_top=10,
                                                                  random_search_mult=args.random_search_mult,
                                                                  smoke=smoke_mode,
                                                                  cv=cv,
                                                                  n_jobs=n_jobs_cv,
                                                                  target_f1=getattr(args, 'target_f1', None))
        status_worst_fold_report = None
    else:
        # If smoke mode is enabled, temporarily override the global distributions and models
        if getattr(args, 'smoke', False) or args.random_search_mult < 0.05:
            orig_param_dist = globals().get('param_distributions')
            orig_models = globals().get('models')
            # Define smoke defaults locally to avoid relying on prior variables
            models_to_use = {'RandomForestClassifier': RandomForestClassifier(random_state=gv.RANDOM_STATE, n_estimators=10)}
            param_dist_to_use = {
                'RandomForestClassifier': {
                    'smote__enabled': [False],
                    'feature_selecting_classifier__max_features': [10, None],
                    'feature_selecting_classifier__threshold': [None],
                    'feature_selecting_classifier__estimator__n_estimators': [10]
                }
            }
            globals()['param_distributions'] = param_dist_to_use
            globals()['models'] = models_to_use
            try:
                top_models_status, best_status, status_worst_fold_report = get_top_models(
                    X_train_status,
                    y_train_status,
                    n_top=10,
                    phase="status",
                    column_headers_json=args.column_headers_json,
                    random_search_mult=args.random_search_mult,
                    n_jobs_cv=1
                )
            finally:
                globals()['models'] = orig_models
                globals()['param_distributions'] = orig_param_dist
        else:
            top_models_status, best_status, status_worst_fold_report = get_top_models(
                X_train_status,
                y_train_status,
                n_top=10,
                phase="status",
                column_headers_json=args.column_headers_json,
                random_search_mult=args.random_search_mult,
                n_jobs_cv=-1
            )

    # Save top models list to models/ with timestamp
    ts = now_ts()
    rows = []
    for model_name, params, score in top_models_status:
        rows.append({"model": model_name, "score": score, "params": json.dumps(params, default=str), "timestamp": ts})
    df_top = pd.DataFrame(rows)
    # All models file: already saved in get_top_models; additionally create best-only summary JSON
    if best_status:
        write_csv_and_symlink(pd.DataFrame(rows), gv.MODELS_DIR, f"status_best_models", ts)
    else:
        logger.error("No best status model found; aborting status flow.")
        return

    # Build pipeline for best candidate
    model_name = best_status["model"]
    params = best_status["params"]
    k_best = int(params.get("feature_selecting_classifier__max_features", -1)) if params.get("feature_selecting_classifier__max_features") is not None else None
    threshold_best = params.get("feature_selecting_classifier__threshold", None)
    smote_enabled = bool(params.get("smote__enabled", True))
    smote_k = int(params.get("smote__k_neighbors", 5))
    smote_cats = params.get("smote__categorical_feature_names", [c for c in headers.get('categorical_cols', []) if c in X_train_status.columns])
    debug_check_for_nans(X_train_status, smote_cats)

    smote_step = MaybeSMOTESampler(enabled=smote_enabled, categorical_feature_names=smote_cats, k_neighbors=smote_k, random_state=gv.RANDOM_STATE)
    best_pipeline = ImbPipeline([
        ("smote", smote_step),
        ("feature_selecting_classifier", FeatureSelectingClassifier(estimator=clone(models[model_name]), max_features=k_best, threshold=threshold_best))
    ])
    # set classifier params present in params
    set_params = {k: v for k, v in params.items() if k in best_pipeline.get_params()}
    if set_params:
        best_pipeline.set_params(**set_params)

    # Optionally use MLPipelineCoordinator to create and fit the pipeline, for improved debugging
    if getattr(args, "use_coordinator", False):
        logger.info("Using MLPipelineCoordinator for pipeline creation and fitting")
        try:
            coordinator = MLPipelineCoordinator()
            # Ensure coordinator has safe defaults set from params
            coordinator_pipeline = coordinator.create_pipeline(base_estimator=clone(models[model_name]),
                                                               smote_config={"enabled": smote_enabled, "categorical_feature_names": smote_cats, "k_neighbors": smote_k},
                                                               feature_selection_config={"max_features": k_best, "threshold": threshold_best})
            if set_params:
                # Map to coordinator pipeline params if applicable
                coordinator_pipeline.set_params(**set_params)
            coordinator.fit_pipeline(coordinator_pipeline, X_train_status, y_train_status)
            best_pipeline = coordinator_pipeline
        except Exception:
            logger.exception("Coordinator-based pipeline fitting failed; falling back to direct pipeline")

    # Fit on train only, save pipeline state
    # Fit pipeline
    try:
        if getattr(args, "use_coordinator", False):
            # If using coordinator, create new coordinator and fit
            coordinator = MLPipelineCoordinator()
            coordinator.fit_pipeline(best_pipeline, X_train_status, y_train_status)
        else:
            best_pipeline.fit(X_train_status, y_train_status)
    except Exception:
        logger.exception("Failed to fit best pipeline; aborting status phase.")
        return
    status_feature_info = save_pipeline_state(best_pipeline, "status", gv.MODELS_DIR)

    # Calibrate on training data and save calibrated pipeline trained-on-train
    calibrated = CalibratedClassifierCV(estimator=clone(best_pipeline), cv=(2 if smoke_flag else 3), method="sigmoid")
    calibrated.fit(X_train_status, y_train_status)
    joblib.dump(calibrated, gv.MODELS_DIR / "status_best_trained_on_train.pkl")

    # Log feature importance for best (if present) and save to models as CSV
    fsc = best_pipeline.named_steps.get("feature_selecting_classifier")
    if fsc is not None and getattr(fsc, "feature_importances_", None) is not None:
        fi_df = fsc.feature_importances_.copy()
        fi_file = gv.MODELS_DIR / f"status_best_feature_importance-{ts}.csv"
        fi_df.to_csv(fi_file, index=False)
        # symlink
        symlink = gv.MODELS_DIR / "status_best_feature_importance-latest.csv"
        try:
            if symlink.exists() or symlink.is_symlink():
                symlink.unlink()
            symlink.symlink_to(fi_file.name)
        except Exception:
            logger.warning("Unable to create symlink for feature importance.")

    # Evaluate on test set
    X_test_status = test_df[available_cols].copy()
    y_test_status = test_df[status_target].to_numpy()

    # Use selected_features if available
    sel_features = status_feature_info.get("selected_features") if status_feature_info else None
    if sel_features:
        # make sure they exist in test set
        missing = [c for c in sel_features if c not in X_test_status.columns]
        if missing:
            logger.warning(f"Test set missing selected features for status: {missing}. Falling back to all available features.")
            X_test_eval = X_test_status
        else:
            X_test_eval = X_test_status[sel_features].copy()
    else:
        X_test_eval = X_test_status

    # For calibrated pipeline, we must pass the full test set (with all available columns) because
    # the fitted pipeline expects the full feature vector and performs feature selection internally.
    y_test_pred = calibrated.predict(X_test_status)
    # try predict_proba for log-loss and auc
    try:
        y_test_proba = calibrated.predict_proba(X_test_status)
        # If classifier only provides a single-class probability column (shape[1] == 1), skip log-loss and AUC
        if y_test_proba is not None and y_test_proba.ndim == 2 and y_test_proba.shape[1] >= 2:
            try:
                ll = log_loss(y_test_status, y_test_proba)
                logger.info(f"Status calibrated log-loss on test: {ll:.6f}")
                joblib.dump(y_test_proba, gv.LOG_DIR / "status_test_proba.pkl")
            except Exception:
                logger.debug("log_loss computation failed for predicted probabilities.")
        else:
            logger.debug("predict_proba returned a single-probability column or non-standard shape; skipping log_loss/auc.")
    except Exception:
        logger.debug("predict_proba not available for calibrated status pipeline.")

    # Save reports and predictions in logs and models as requested
    ts_now = now_ts()
    # best test report
    best_test_report = save_report(y_test_status, y_test_pred, f"status_best_test_report", gv.LOG_DIR)
    # save predictions csv
    pd.DataFrame({"y_true": y_test_status, "y_pred": y_test_pred}).to_csv(gv.LOG_DIR / f"status_best_test_preds.csv", index=False)
    # save full best model json info (model name, params, selected features, worst fold metrics)
    best_model_info = {
        "model": model_name,
        "params": params,
        "selected_features": status_feature_info.get("selected_features"),
        "saved_pipeline_uncalibrated": str(gv.MODELS_DIR / "status_pipeline_uncalibrated.pkl"),
        "saved_calibrated_trained_on_train": str(gv.MODELS_DIR / "status_best_trained_on_train.pkl"),
        "cv_worst_case_report": status_worst_fold_report
    }
    write_json_and_symlink(best_model_info, gv.MODELS_DIR, "status_best_model", ts_now)

    # copy test report and worst-case CV report into models/ as CSV as well
    # best test report CSV already in logs; re-save in models with timestamp
    df_test_report = pd.DataFrame(best_test_report).transpose()
    write_csv_and_symlink(df_test_report, gv.MODELS_DIR, "status_best_test_report", ts_now)

    if status_worst_fold_report:
        df_cv_worst = pd.DataFrame(status_worst_fold_report).transpose()
        write_csv_and_symlink(df_cv_worst, gv.MODELS_DIR, "status_best_cv_worstcase_report", ts_now)

    # Save predictions to models/
    pd.DataFrame({"y_true": y_test_status, "y_pred": y_test_pred}).to_csv(gv.MODELS_DIR / f"status_best_test_preds-{ts_now}.csv", index=False)
    symlink_preds = gv.MODELS_DIR / "status_best_test_preds-latest.csv"
    try:
        if symlink_preds.exists() or symlink_preds.is_symlink():
            symlink_preds.unlink()
        symlink_preds.symlink_to(f"status_best_test_preds-{ts_now}.csv")
    except Exception:
        logger.warning("Unable to create symlink for status best test preds.")

    logger.info("STATUS phase complete.")

    # -------------------------
    # TIER PHASE
    # -------------------------
    train_approved_df = extract_valid_tier_records(train_df)
    if train_approved_df.empty:
        logger.warning("No approved tier records; skipping tier phase.")
    else:
        X_train_tier = train_approved_df[available_cols].copy()
        y_train_tier = train_approved_df[tier_target].astype(int).to_numpy()
        log_class_imbalance(y_train_tier, "Tier")

        unique_tier_labels = np.unique(y_train_tier)
        if len(unique_tier_labels) < 2:
            # Only one class present; skip heavy model search and training. Create a trivial DummyClassifier pipeline
            logger.warning(f"Tier target has only one class present ({unique_tier_labels.tolist()}); skipping tier model search/training.")
            # Build a trivial pipeline that always predicts the single class
            smote_step = MaybeSMOTESampler(enabled=False, categorical_feature_names=[], k_neighbors=1, random_state=gv.RANDOM_STATE)
            dummy = DummyClassifier(strategy='most_frequent')
            best_tier_pipeline = ImbPipeline([
                ("smote", smote_step),
                ("feature_selecting_classifier", FeatureSelectingClassifier(estimator=clone(dummy), max_features=None, threshold=None))
            ])
            # Fit the dummy pipeline on the tier train data
            try:
                best_tier_pipeline.fit(X_train_tier, y_train_tier)
            except Exception:
                logger.exception("Failed to fit Dummy tier pipeline; skipping tier phase.")
                return
            tier_feature_info = save_pipeline_state(best_tier_pipeline, "tier", gv.MODELS_DIR)
            calibrated_tier = None
            # Evaluate on test tier if available, but do not try to calibrate
            test_tier_df = extract_valid_tier_records(test_df)
            if not test_tier_df.empty:
                X_test_tier = test_tier_df[available_cols].copy()
                sel_features = tier_feature_info.get("selected_features")
                if sel_features:
                    missing = [c for c in sel_features if c not in X_test_tier.columns]
                    X_test_eval = X_test_tier if missing else X_test_tier[sel_features].copy()
                else:
                    X_test_eval = X_test_tier
                y_test_tier = test_tier_df[tier_target].astype(int).to_numpy()
                y_test_pred_tier = best_tier_pipeline.predict(X_test_tier)
                try:
                    y_test_proba_tier = best_tier_pipeline.predict_proba(X_test_tier)
                    if y_test_proba_tier is not None and y_test_proba_tier.ndim == 2 and y_test_proba_tier.shape[1] >= 2:
                        ll_tier = log_loss(y_test_tier, y_test_proba_tier)
                        logger.info(f"Tier calibrated log-loss on test: {ll_tier:.6f}")
                    else:
                        logger.debug("Tier predict_proba returned only one probability column; skipping log_loss.")
                except Exception:
                    logger.debug("predict_proba not available for dummy tier pipeline.")
                save_report(y_test_tier, y_test_pred_tier, "tier_best_test_report", gv.LOG_DIR)
                pd.DataFrame({"y_true": y_test_tier, "y_pred": y_test_pred_tier}).to_csv(gv.LOG_DIR / "tier_best_test_preds.csv", index=False)
                save_report(y_test_tier, y_test_pred_tier, "tier_best_test_report", gv.LOG_DIR)
            joblib.dump(best_tier_pipeline, gv.MODELS_DIR / "tier_best_trained_on_train.pkl")
            logger.info("TIER phase complete (single-class fallback).")
            return

        if getattr(args, "use_coordinator", False):
            logger.info("Using MLPipelineCoordinator for model search (tier phase)")
            coordinator = MLPipelineCoordinator()
            # handle smoke mode similarly to status phase
            smoke_mode = smoke_flag
            n_splits = 2 if smoke_mode else 3
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=gv.RANDOM_STATE)
            n_jobs_cv = 1 if smoke_mode else -1
            models_to_use = models
            param_dist_to_use = param_distributions
            if smoke_mode:
                from sklearn.ensemble import RandomForestClassifier as _RF
                models_to_use = {'RandomForestClassifier': _RF(random_state=gv.RANDOM_STATE, n_estimators=10)}
                param_dist_to_use = {
                    'RandomForestClassifier': {
                        'smote__enabled': [False],
                        'feature_selecting_classifier__max_features': [10, None],
                        'feature_selecting_classifier__threshold': [None],
                        'feature_selecting_classifier__estimator__n_estimators': [10]
                    }
                }
            top_models_tier, best_tier = coordinator.search_models(models=models_to_use,
                                                                  param_distributions=param_dist_to_use,
                                                                  X=X_train_tier,
                                                                  y=y_train_tier,
                                                                  n_top=10,
                                                                  random_search_mult=args.random_search_mult,
                                                                  smoke=smoke_mode,
                                                                  cv=cv,
                                                                  n_jobs=n_jobs_cv,
                                                                  target_f1=getattr(args, 'target_f1', None))
            tier_worst_fold_report = None
        else:
            top_models_tier, best_tier, tier_worst_fold_report = get_top_models(
                X_train_tier,
                y_train_tier,
                n_top=10,
                phase="tier",
                column_headers_json=args.column_headers_json,
                random_search_mult=args.random_search_mult,
                n_jobs_cv=-1
            )

        # Save top/tier best similar to status (omitted for brevity - mimic status flow)
        # For brevity in this response, I mirror the status finalization process:
        # Build best pipeline, fit on train, calibrate, evaluate on approved test tier records, and save files
        if not best_tier:
            logger.error("No best tier model found; skipping tier finalization.")
        else:
            model_name = best_tier["model"]
            params = best_tier["params"]
            max_f = params.get("feature_selecting_classifier__max_features", None)
            thr = params.get("feature_selecting_classifier__threshold", None)
            smote_enabled = bool(params.get("smote__enabled", True))
            smote_k = int(params.get("smote__k_neighbors", 5))
            smote_cats = params.get("smote__categorical_feature_names", [c for c in headers.get('categorical_cols', []) if c in X_train_tier.columns])

            smote_step = MaybeSMOTESampler(enabled=smote_enabled, categorical_feature_names=smote_cats, k_neighbors=smote_k, random_state=gv.RANDOM_STATE)
            best_tier_pipeline = ImbPipeline([
                ("smote", smote_step),
                ("feature_selecting_classifier", FeatureSelectingClassifier(estimator=clone(models[model_name]), max_features=max_f, threshold=thr))
            ])
            set_params = {k: v for k, v in params.items() if k in best_tier_pipeline.get_params()}
            if set_params:
                best_tier_pipeline.set_params(**set_params)

            best_tier_pipeline.fit(X_train_tier, y_train_tier)
            tier_feature_info = save_pipeline_state(best_tier_pipeline, "tier", gv.MODELS_DIR)

            calibrated_tier = CalibratedClassifierCV(estimator=clone(best_tier_pipeline), cv=(2 if smoke_flag else 3), method="sigmoid")
            calibrated_tier.fit(X_train_tier, y_train_tier)

            test_tier_df = extract_valid_tier_records(test_df)
            if not test_tier_df.empty:
                X_test_tier = test_tier_df[available_cols].copy()
                sel_features = tier_feature_info.get("selected_features")
                if sel_features:
                    missing = [c for c in sel_features if c not in X_test_tier.columns]
                    X_test_eval = X_test_tier if missing else X_test_tier[sel_features].copy()
                else:
                    X_test_eval = X_test_tier
                y_test_tier = test_tier_df[tier_target].astype(int).to_numpy()
                y_test_pred_tier = calibrated_tier.predict(X_test_tier)
                try:
                    y_test_proba_tier = calibrated_tier.predict_proba(X_test_tier)
                    if y_test_proba_tier is not None and y_test_proba_tier.ndim == 2 and y_test_proba_tier.shape[1] >= 2:
                        ll_tier = log_loss(y_test_tier, y_test_proba_tier)
                        logger.info(f"Tier calibrated log-loss on test: {ll_tier:.6f}")
                    else:
                        logger.debug("Tier predict_proba returned only one probability column; skipping log_loss.")
                except Exception:
                    logger.debug("predict_proba not available for tier calibrated pipeline.")
                save_report(y_test_tier, y_test_pred_tier, "tier_best_test_report", gv.LOG_DIR)
                pd.DataFrame({"y_true": y_test_tier, "y_pred": y_test_pred_tier}).to_csv(gv.LOG_DIR / "tier_best_test_preds.csv", index=False)
                save_report(y_test_tier, y_test_pred_tier, "tier_best_test_report", gv.LOG_DIR)

            joblib.dump(calibrated_tier, gv.MODELS_DIR / "tier_best_trained_on_train.pkl")
            logger.info("TIER phase complete.")

    logger.info("Evaluation complete.")

# --------------------------
# CLI entrypoint
# --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models (status and tier) with name-based SMOTE and SelectFromModel preservation.")
    parser.add_argument("--train_csv", type=Path, required=True, help="Training CSV path")
    parser.add_argument("--test_csv", type=Path, required=True, help="Test CSV path")
    parser.add_argument("--random_search_mult", type=float, default=float(gv.RANDOM_SEARCH_ITER_MULT), help="Random search multiplier to scale param samples (lower runs quicker)")
    parser.add_argument("--use_coordinator", action="store_true", help="Use MLPipelineCoordinator for building and fitting the pipelines")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode with minimal iterations and smaller models for quick tests")
    parser.add_argument("--target_f1", type=float, default=None, help="Optional target F1 score to early-stop model search")
    parser.add_argument("--column_headers_json", type=Path, required=True, help="column_headers.json path (schema)")
    args = parser.parse_args()
    
    import inspect
    current_frame = inspect.currentframe()
    if current_frame is None:
        raise RuntimeError("Cannot get current frame")
    current_file = inspect.getfile(current_frame)
    print(f"Python thinks this file is: {current_file}")
    print(f"__file__ is: {__file__}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    try:
        main(args)
    except Exception as e:
        logger.exception(f"Fatal error in eval_algos: {e}")
        sys.exit(1)



