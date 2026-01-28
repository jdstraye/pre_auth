#!/usr/bin/env python3
"""Clean exhaustive hyperparameter search runner.

Performs a small safe search by default, captures per-fold `cv_fold_info`, and
ensures the progress bar is closed in all cases.
"""
from pathlib import Path
import argparse
import json
import logging
import time
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn.metrics import f1_score
from sklearn.base import clone

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from src.utils import load_column_headers, gv, sanitize_column_name
from src.ingest import _load_golden_schema, _parse_schema, preprocess_dataframe
from src.pipeline_coordinator import MLPipelineCoordinator
from src.eval_algos import param_distributions as eval_param_distributions, models as eval_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def param_random_samples(dist, n_samples, random_state):
    try:
        from src.eval_utils import generate_parameter_samples
        return generate_parameter_samples(dist, n_samples=n_samples, random_state=random_state)
    except Exception:
        return list(ParameterSampler(dist, n_iter=max(1, int(n_samples)), random_state=random_state))


def run_search(args):
    gv.DEBUG_MODE = bool(args.debug)

    df = pd.read_csv(args.data_csv)
    df.columns = [sanitize_column_name(c) for c in df.columns]
    schema = _load_golden_schema(Path(args.column_headers))
    sorted_schema, column_map = _parse_schema(schema)
    df = preprocess_dataframe(df, sorted_schema, column_map)

    headers = load_column_headers(Path(args.column_headers), df)
    feature_cols = [c for c in headers.get('feature_cols', []) if c in df.columns]
    target_cols = [c for c in headers.get('target_cols', []) if c in df.columns]
    if not target_cols:
        raise ValueError('No target columns present')
    status_target = next((t for t in target_cols if t.lower().endswith('status_label')), target_cols[0])

    X = df[feature_cols].copy()
    y = df[status_target].astype(int).to_numpy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=gv.RANDOM_STATE)

    coordinator = MLPipelineCoordinator(enable_debugging=args.debug, export_debug_info=args.debug)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_candidate_grids = {}
    total_candidates = 0
    for mname, dist in eval_param_distributions.items():
        if args.exhaustive:
            samples = list(ParameterGrid(dist))
        else:
            n = args.n_samples_per_model or 10
            samples = param_random_samples(dist, n, random_state=gv.RANDOM_STATE)
        model_candidate_grids[mname] = samples
        total_candidates += len(samples)

    logger.info(f"Total parameter combinations: {total_candidates}")

    records = []
    candidate_counter = 0
    pbar = tqdm(total=total_candidates, desc='Search', unit='cand', dynamic_ncols=True) if (not args.no_progress and HAS_TQDM) else None

    try:
        for mname, model in eval_models.items():
            combos = model_candidate_grids.get(mname, [])
            if args.limit and args.limit > 0:
                combos = combos[:args.limit]
            for idx, cand in enumerate(combos):
                candidate_counter += 1
                if pbar is not None:
                    pbar.update(1)

                try:
                    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=gv.RANDOM_STATE)
                    fold_metrics = []
                    cv_fold_info = []
                    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full)):
                        X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
                        y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]
                        # Build pipeline with SMOTE and feature-selection configs derived from candidate
                        smote_cfg = {"enabled": cand.get('smote__enabled', True), "categorical_feature_names": cand.get('smote__categorical_feature_names', []), "k_neighbors": int(cand.get('smote__k_neighbors', 5))}
                        feat_cfg = {"max_features": cand.get('feature_selecting_classifier__max_features', None), "threshold": cand.get('feature_selecting_classifier__threshold', None)}
                        pipeline = coordinator.create_pipeline(base_estimator=clone(model), smote_config=smote_cfg, feature_selection_config=feat_cfg)
                        try:
                            pipeline.set_params(**cand)
                        except Exception:
                            pass
                        fitted = coordinator.fit_pipeline(pipeline, X_tr, y_tr)
                        y_pred = fitted.predict(X_val)
                        f1 = float(f1_score(y_val, y_pred, average='macro'))
                        fold_metrics.append({'f1': f1})
                        cv_fold_info.append({'fold': int(fold_i), 'f1': float(f1)})

                    mean_f1 = float(np.mean([m['f1'] for m in fold_metrics])) if fold_metrics else 0.0

                    pipeline_full = coordinator.create_pipeline(base_estimator=clone(model), smote_config=smote_cfg, feature_selection_config=feat_cfg)
                    try:
                        pipeline_full.set_params(**cand)
                    except Exception:
                        pass
                    pipeline_full.fit(X_train_full, y_train_full)
                    y_pred_test = pipeline_full.predict(X_test)
                    test_f1 = float(f1_score(y_test, y_pred_test, average='macro'))

                    records.append({'timestamp': time.time(), 'model': mname, 'params': json.dumps(cand, default=str), 'mean_f1': mean_f1, 'test_f1': test_f1, 'cv_fold_info': json.dumps(cv_fold_info)})

                    if args.flush_every > 0 and (candidate_counter % args.flush_every == 0):
                        pd.DataFrame(records).to_csv(outdir / 'search_results.csv', mode='a', index=False, header=not (outdir / 'search_results.csv').exists())
                        records.clear()

                except Exception as e:
                    logger.exception(f"Candidate evaluation failed: {e}")
                    continue

    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    if records:
        pd.DataFrame(records).to_csv(outdir / 'search_results.csv', mode='a', index=False, header=not (outdir / 'search_results.csv').exists())

    out_csv = outdir / 'search_results.csv'
    df_out = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    print(f"[DEBUG] Output CSV: {out_csv}")
    print(f"[DEBUG] DataFrame columns: {list(df_out.columns)}")
    print(f"[DEBUG] DataFrame head:\n{df_out.head()}")
    if not df_out.empty:
        if 'mean_f1' in df_out.columns:
            df_out.sort_values(by='mean_f1', ascending=False, inplace=True)
            (outdir / 'top_candidates.json').write_text(df_out.head(args.n_top).to_json(orient='records'))
        else:
            print("[ERROR] 'mean_f1' column missing from search_results.csv. Candidates may have failed or output is malformed.")


def main():
    parser = argparse.ArgumentParser(description='Robust exhaustive hyperparameter search')
    parser.add_argument('--data-csv', type=Path, default=ROOT / 'data' / 'prefi_weaviate_clean-1_flattened.csv')
    parser.add_argument('--column-headers', type=Path, default=ROOT / 'src' / 'column_headers.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models' / 'exhaustive_search')
    parser.add_argument('--n-top', type=int, default=5)
    parser.add_argument('--n-samples-per-model', type=int, default=10)
    parser.add_argument('--flush-every', type=int, default=10)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--exhaustive', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    run_search(args)


if __name__ == '__main__':
    main()
