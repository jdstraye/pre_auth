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
from src.ingest import _load_golden_schema, _parse_schema, preprocess_dataframe, parse_json
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

    # Support either CSV input or a direct JSON payload (tests call with --input-json)
    schema = _load_golden_schema(Path(args.column_headers))
    sorted_schema, column_map = _parse_schema(schema)

    if getattr(args, 'input_json', None) is not None:
        # Parse and preprocess the provided JSON input
        df = parse_json(Path(args.input_json), sorted_schema)
        df = preprocess_dataframe(df, sorted_schema, column_map)
    else:
        df = pd.read_csv(args.data_csv)
        df.columns = [sanitize_column_name(c) for c in df.columns]
        df = preprocess_dataframe(df, sorted_schema, column_map)

    headers = load_column_headers(Path(args.column_headers), df)
    feature_cols = [c for c in headers.get('feature_cols', []) if c in df.columns]
    target_cols = [c for c in headers.get('target_cols', []) if c in df.columns]
    if not target_cols:
        raise ValueError('No target columns present')
    status_target = next((t for t in target_cols if t.lower().endswith('status_label')), target_cols[0])

    X = df[feature_cols].copy()
    y = df[status_target].astype(int).to_numpy()

    # Print mapping of feature indices to column names for inspection
    print("\nFeature index-to-name mapping for current classifier:")
    for idx, col in enumerate(feature_cols):
        print(f"feature_{idx}: {col}")

    # Example: Print summary statistics and correlation for first three features (customize as needed)
    if len(feature_cols) >= 3:
        print("\nSample statistics for first three features:")
        print(X[[feature_cols[0], feature_cols[1], feature_cols[2]]].describe())
        print("\nCorrelation matrix for first three features:")
        print(X[[feature_cols[0], feature_cols[1], feature_cols[2]]].corr())

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=gv.RANDOM_STATE)

    coordinator = MLPipelineCoordinator(enable_debugging=args.debug, export_debug_info=args.debug)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Ensure previous search results are cleared to avoid leftover files with malformed headers
    out_csv = outdir / 'search_results.csv'
    if out_csv.exists():
        try:
            out_csv.unlink()
        except Exception:
            pass

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
                    # Record failure so output file is produced and includes mean_f1 column
                    records.append({'timestamp': time.time(), 'model': mname, 'params': json.dumps(cand, default=str), 'mean_f1': float('nan'), 'test_f1': float('nan'), 'cv_fold_info': json.dumps({'error': str(e)})})
                    if args.flush_every > 0 and (candidate_counter % args.flush_every == 0):
                        pd.DataFrame(records).to_csv(outdir / 'search_results.csv', mode='a', index=False, header=not (outdir / 'search_results.csv').exists())
                        records.clear()
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
    if not df_out.empty and 'mean_f1' in df_out.columns:
        df_out.sort_values(by='mean_f1', ascending=False, inplace=True)
        n_top = getattr(args, 'n_top', 5)
        (outdir / 'top_candidates.json').write_text(df_out.head(n_top).to_json(orient='records'))
    elif not df_out.empty:
        logging.warning(f"search_results.csv exists but 'mean_f1' column is missing. Columns: {df_out.columns.tolist()}")
    else:
        logging.warning("search_results.csv does not exist or is empty. No candidates to rank.")

    # ----------------------
    # Snapshot results + run history appending (to configurable reports dir)
    # ----------------------
    try:
        ts_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())

        # Determine reports dir (configurable). Default: ROOT/output/reports
        reports_dir = Path(getattr(args, 'reports_dir', ROOT / 'output' / 'reports'))
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception(f"Failed to create reports_dir: {reports_dir}")

        # If results file exists, copy to timestamped CSV in reports_dir and update latest symlink
        if out_csv.exists():
            import shutil
            dest = reports_dir / f"search_results_{ts_str}.csv"
            try:
                shutil.copy2(out_csv, dest)
            except Exception:
                # Fall back to pandas if copy fails for any reason
                pd.read_csv(out_csv).to_csv(dest, index=False)

            latest = reports_dir / 'search_results_latest.csv'
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                # create a relative symlink to the timestamped file name
                latest.symlink_to(dest.name)
            except Exception:
                logger.exception("Failed to create/update latest search_results symlink")

            # Also keep a copy of the most recent search_results.csv next to outdir for backward compatibility
            try:
                pd.read_csv(dest).to_csv(outdir / 'search_results.csv', index=False)
            except Exception:
                logger.exception("Failed to ensure outdir search_results.csv copy is present")

            # Copy top_candidates JSON into reports_dir with timestamp and update latest symlink
            try:
                top_src = outdir / 'top_candidates.json'
                if top_src.exists():
                    top_dest = reports_dir / f"top_candidates_{ts_str}.json"
                    import shutil as _sh
                    try:
                        _sh.copy2(top_src, top_dest)
                    except Exception:
                        top_src_text = top_src.read_text()
                        top_dest.write_text(top_src_text)
                    top_latest = reports_dir / 'top_candidates_latest.json'
                    if top_latest.exists() or top_latest.is_symlink():
                        try:
                            top_latest.unlink()
                        except Exception:
                            pass
                    try:
                        top_latest.symlink_to(top_dest.name)
                    except Exception:
                        logger.exception("Failed to create/update top_candidates_latest symlink")
            except Exception:
                logger.exception("Failed to copy top_candidates to reports dir")

        # Append a run summary block to RUN_HISTORY.md in the reports dir
        try:
            history_file = reports_dir / 'RUN_HISTORY.md'
            # Build args map (safe string conversion)
            args_map = {}
            for k in dir(args):
                if k.startswith('_'):
                    continue
                v = getattr(args, k)
                if callable(v) or isinstance(v, (type,)):
                    continue
                try:
                    args_map[k] = str(v)
                except Exception:
                    args_map[k] = repr(v)

            # Compute per-model success counts and top candidates
            n_top_write = getattr(args, 'n_top', 5)
            zero_models = []
            top_rows = pd.DataFrame()
            if (reports_dir / f"search_results_{ts_str}.csv").exists():
                try:
                    df_snap = pd.read_csv(reports_dir / f"search_results_{ts_str}.csv")
                    df_snap['mean_f1'] = pd.to_numeric(df_snap['mean_f1'], errors='coerce')
                    model_counts = df_snap.groupby('model')['mean_f1'].apply(lambda s: s.notna().sum()).to_dict()
                    zero_models = [m for m, c in model_counts.items() if c == 0]
                    df_sorted = df_snap.sort_values(by='mean_f1', ascending=False)
                    top_rows = df_sorted.head(n_top_write)
                except Exception:
                    logger.exception("Failed to parse timestamped search_results.csv for history summary")

            # Write the RUN_HISTORY.md entry
            lines = []
            lines.append(f"## {ts_str}\n")
            lines.append(f"- args: {json.dumps(args_map)}\n")
            lines.append(f"- models_present: {json.dumps(list(eval_models.keys()))}\n")
            if zero_models:
                lines.append(f"- models_with_no_valid_candidates: {json.dumps(zero_models)}\n")
            if (reports_dir / f"search_results_{ts_str}.csv").exists():
                lines.append(f"- search_results_file: { (reports_dir / f'search_results_{ts_str}.csv').name }\n")
                lines.append(f"- search_results_latest: search_results_latest.csv\n")
            lines.append(f"- top_{n_top_write}:\n")
            for _, row in top_rows.iterrows():
                lines.append(f"  - model: {row.get('model')}, mean_f1: {row.get('mean_f1')}, test_f1: {row.get('test_f1')}\n")
                lines.append(f"    params: {row.get('params')}\n")
            lines.append("\n")

            with history_file.open('a') as fh:
                fh.write('\n'.join(lines))

        except Exception:
            logger.exception("Failed to append RUN_HISTORY.md")
    except Exception:
        logger.exception("Failed to snapshot search results or write run history")


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
    parser.add_argument('--reports-dir', type=Path, default=ROOT / 'output' / 'reports', help='Directory to persist search snapshots and run history')
    args = parser.parse_args()
    run_search(args)


if __name__ == '__main__':
    main()
