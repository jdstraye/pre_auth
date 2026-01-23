"""Promote auto-mapped GT (tmp/auto_map/*.mapped.json) into validated ground-truth JSONs.

Features
- Accepts a mapped JSON (output of `scripts/auto_map_unvalidated.py`) and promotes accepted factors
  into `data/extracted/<name>_ground_truth.json` with required fields (`factor,page,bbox,spans,canonical_key`).
- Supports `--accept-all` (promote all matched rows), `--accept-indices` (comma-separated factor indices),
  or `--csv` (CSV with mapped_json,accept boolean per-row).
- Safety: has `--dry-run` (no file writes), `--backup` (save original unvalidated to .bak), and runs
  `scripts/validate_ground_truth.py` on the output before allowing commit.
- Optional `--commit` will `git add` and `git commit` the new file (no push).

Usage examples
- Promote everything matched with score >= 0.65 (dry-run):
    PYTHONPATH=. python3 scripts/promote_validated_gt.py tmp/auto_map/user_1314_...mapped.json --accept-all --dry-run

- Promote specific factor indices (0-based):
    PYTHONPATH=. python3 scripts/promote_validated_gt.py tmp/auto_map/user_1131...mapped.json --accept-indices 0,2,5 --commit

- Promote by CSV of reviewer decisions and commit (safe):
    PYTHONPATH=. python3 scripts/promote_validated_gt.py --csv regression/review_decisions.csv --commit

Return codes
- 0 on success (or dry-run ok)
- non-zero on validation or other failures
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

# Allow running as `python scripts/promote_validated_gt.py` without setting PYTHONPATH explicitly
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = Path('regression/quick50_manifest.json')
VALIDATOR = Path('scripts/validate_ground_truth.py')


def load_mapped(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_validated_payload(mapped: Dict[str, Any], accept_indices: Set[int]) -> Dict[str, Any]:
    # Reconstruct a GT JSON compatible with existing ground-truth files
    # Use mapped['source_gt'] to copy metadata where possible
    src_gt = Path(mapped.get('source_gt', ''))
    out: Dict[str, Any] = {}
    # try to preserve top-level fields from original unvalidated if present
    if src_gt.exists():
        orig = json.loads(src_gt.read_text(encoding='utf-8'))
        for k in ('filename', 'source', 'credit_score', 'age'):
            if k in orig:
                out[k] = orig[k]
    # credit_factors
    out['credit_factors'] = []
    for idx, row in enumerate(mapped.get('mapped', [])):
        if idx not in accept_indices:
            continue
        if row.get('match_type') == 'none' and not row.get('spans'):
            # cannot promote unmapped row without spans/bbox
            raise ValueError(f"Row {idx} has no mapping/spans and cannot be promoted: {row.get('factor')}")
        factor_obj = {
            'factor': row.get('factor'),
            'color': row.get('color'),
            'hex': row.get('color') if row.get('color','').startswith('#') else row.get('hex', row.get('color')),
            'page': row.get('page'),
            'bbox': row.get('bbox'),
            'spans': row.get('spans'),
            'canonical_key': row.get('canonical_key') or ''
        }
        out['credit_factors'].append(factor_obj)
    return out


def run_validator(path: Path) -> None:
    # returns None or raises
    if not VALIDATOR.exists():
        raise FileNotFoundError(f"validator script missing: {VALIDATOR}")
    res = subprocess.run(['python3', str(VALIDATOR), str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Validator failed for {path}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")


def update_manifest(validated_path: Path) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    entry = {'path': str(validated_path), 'filename': validated_path.name}
    if entry not in manifest:
        manifest.append(entry)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def git_commit(paths: List[Path], message: str) -> None:
    subprocess.check_call(['git', 'add'] + [str(p) for p in paths])
    subprocess.check_call(['git', 'commit', '-m', message])


def parse_accept_csv(path: Path) -> Dict[str, List[int]]:
    # CSV expected: mapped_json,accept_indices (comma-separated)
    out: Dict[str, List[int]] = {}
    with path.open(encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            mj = row.get('mapped_json')
            idxs = row.get('accept_indices', '')
            if not mj:
                continue
            out[mj] = [int(x) for x in idxs.split(',') if x.strip()]
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapped', help='single mapped JSON (tmp/auto_map/...)')
    ap.add_argument('--csv', help='review CSV with mapped_json and accept_indices columns')
    ap.add_argument('--accept-all', action='store_true', help='accept all matched rows in the mapped JSON')
    ap.add_argument('--accept-indices', help='comma-separated 0-based indices to accept (single mapped JSON only)')
    ap.add_argument('--min-score', type=float, default=0.65, help='minimum match_score to auto-accept when --accept-all used')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--backup', action='store_true', help='move original unvalidated GT to .bak (git-safe)')
    ap.add_argument('--commit', action='store_true', help='git add & commit the promoted file')
    ap.add_argument('--message', default=None, help='commit message (if --commit)')
    args = ap.parse_args(argv)

    tasks: List[Tuple[Path, Set[int]]] = []
    if args.csv:
        decisions = parse_accept_csv(Path(args.csv))
        for mj, idxs in decisions.items():
            tasks.append((Path(mj), set(idxs)))
    elif args.mapped:
        mjp = Path(args.mapped)
        if args.accept_all:
            mj = json.loads(mjp.read_text(encoding='utf-8'))
            accept = set()
            for i, r in enumerate(mj.get('mapped', [])):
                try:
                    score = float(r.get('match_score') or 0.0)
                except Exception:
                    score = 0.0
                if score >= args.min_score and r.get('match_type') in ('exact', 'substring', 'fuzzy'):
                    accept.add(i)
            tasks.append((mjp, accept))
        elif args.accept_indices:
            idxs = {int(x) for x in args.accept_indices.split(',') if x.strip()}
            tasks.append((mjp, idxs))
        else:
            raise SystemExit('must specify --accept-all or --accept-indices when promoting a single mapped JSON')
    else:
        raise SystemExit('must provide --mapped or --csv')

    promoted_paths: List[Path] = []
    for mj_path, accept_idxs in tasks:
        mapped = load_mapped(mj_path)
        src_gt = Path(mapped.get('source_gt', ''))
        if not src_gt.exists():
            raise FileNotFoundError(f'source GT not found for {mj_path}: {src_gt}')
        payload = build_validated_payload(mapped, accept_idxs)
        # sanitized output path
        out_path = src_gt.with_name(src_gt.name.replace('_ground_truth_unvalidated.json', '_ground_truth.json'))
        print(f"Promoting -> {out_path}  (accepted {len(payload['credit_factors'])} factors)")
        if args.dry_run:
            print('dry-run: skipping write and validation')
            continue
        # backup
        if args.backup:
            bak = src_gt.with_suffix(src_gt.suffix + '.bak')
            if not bak.exists():
                bak.write_text(src_gt.read_text(encoding='utf-8'), encoding='utf-8')
        # write
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        # validate
        try:
            run_validator(out_path)
        except Exception as exc:
            out_path.unlink(missing_ok=True)
            raise
        # update manifest
        update_manifest(out_path)
        promoted_paths.append(out_path)

    if args.commit and promoted_paths:
        msg = args.message or f"data: validate {', '.join(p.name for p in promoted_paths)} GT(s)"
        git_commit(promoted_paths + [MANIFEST], msg)
        print('Committed:', promoted_paths)
    print('Done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
