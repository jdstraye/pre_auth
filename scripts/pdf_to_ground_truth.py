"""Convert PDF(s) to a ground-truth-like JSON file.

By default this emits a text-only GT matching the schema used in `data/extracted/*_ground_truth.json`.
Optionally `--include-spans` will attach page/bbox/spans/canonical_key using the auto-mapper workflow.

Usage:
  python scripts/pdf_to_ground_truth.py data/pdf_analysis/user_1131_credit_summary_2025-09-01_132805.pdf --include-spans --dry-run
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys

# ensure package root is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scripts.pdf_color_extraction import extract_pdf_all_fields
from scripts.auto_map_unvalidated import map_file


def default_out_path(pdf_path: Path) -> Path:
    # derive user id prefix from filename
    stem = pdf_path.stem
    now = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
    out_name = f"{stem}_ground_truth_unvalidated.json"
    return Path('data/extracted') / out_name


def build_text_only_gt(rec: dict, include_spans: bool = False) -> dict:
    # rec may be the top-level dict returned by extract_pdf_all_fields
    # if it contains a .get('rec'), prefer that
    if 'rec' in rec and isinstance(rec['rec'], dict):
        source = rec['rec']
    else:
        source = rec
    out = {}
    # copy top-level known fields
    for k in ('filename','source','credit_score','credit_score_color','age','address','collections_open','collections_closed','public_records','revolving_open_count','revolving_open_total','installment_open_count','installment_open_total','inquiries_last_6_months','monthly_payments','real_estate_open_count','real_estate_open_total','late_pays_2yr','late_pays_gt2yr','red_credit_factors_count','green_credit_factors_count','black_credit_factors_count','credit_freeze','fraud_alert','deceased'):
        if k in source:
            out[k] = source[k]
            # when spans were requested, include any attached bbox/page/spans for these top-level fields
            if include_spans:
                for suff in ('_bbox','_page','_spans'):
                    sk = f"{k}{suff}"
                    if sk in source:
                        out[sk] = source[sk]
    # canonicalize credit_factors to minimal shape
    cfs = source.get('credit_factors', [])
    out['credit_factors'] = []
    for cf in cfs:
        out_cf = {'factor': cf.get('factor')}
        # preserve raw hex when present
        if 'hex' in cf and isinstance(cf.get('hex'), str) and cf.get('hex'):
            # if the extractor mistakenly put a canonical color name into 'hex', normalize
            if not cf.get('hex').startswith('#') and cf.get('hex') in ('red','green','black','neutral','amber'):
                out_cf['color'] = cf.get('hex')
                out_cf['hex'] = None
            else:
                out_cf['hex'] = cf.get('hex')
        # prefer explicit color if provided
        if 'color' in cf and cf.get('color') is not None:
            out_cf['color'] = cf.get('color')
        # derive color from hex or spans if missing
        if 'color' not in out_cf:
            try:
                from src.scripts.pdf_color_extraction import hex_to_rgb, map_color_to_cat
                if out_cf.get('hex'):
                    rgb = hex_to_rgb(out_cf.get('hex'))
                    if rgb:
                        out_cf['color'] = map_color_to_cat(rgb)
                elif include_spans and cf.get('spans'):
                    for s in cf.get('spans'):
                        if s.get('rgb'):
                            out_cf['color'] = map_color_to_cat(tuple(s.get('rgb')))
                            break
                        if s.get('hex'):
                            rgb = hex_to_rgb(s.get('hex'))
                            if rgb:
                                out_cf['color'] = map_color_to_cat(rgb)
                                break
            except Exception:
                # non-fatal: leave color missing
                pass
        # copy bbox/page/spans/canonical_key if available from extractor
        for key in ('bbox','page','spans','canonical_key'):
            if key in cf:
                out_cf[key] = cf[key]
        out['credit_factors'].append(out_cf)

    # also copy any top-level color keys present in the source (e.g., monthly_payments_color)
    for sk in source:
        if sk.endswith('_color') and sk not in out:
            out[sk] = source[sk]

    return out


def attach_spans_to_gt(gt_json_path: Path, pdf_path: Path) -> Path:
    # use the existing auto-mapper: write a temporary unvalidated input and call map_file
    tmp_unvalidated = gt_json_path.with_suffix('.tmp_unvalidated.json')
    shutil.copy(gt_json_path, tmp_unvalidated)
    mapped_json, rows = map_file(str(tmp_unvalidated))
    # read original GT and enrich factors
    gt = json.loads(gt_json_path.read_text(encoding='utf-8'))
    for i, row in enumerate(rows):
        # map by factor text
        ftext = row.get('factor')
        for cf in gt.get('credit_factors', []):
            if cf.get('factor','').strip() == ftext.strip():
                if row.get('page') is not None:
                    cf['page'] = row.get('page')
                if row.get('bbox') is not None:
                    cf['bbox'] = row.get('bbox')
                if row.get('spans') is not None:
                    cf['spans'] = row.get('spans')
                if row.get('canonical_key'):
                    cf['canonical_key'] = row.get('canonical_key')
                break

    # --- Attach spans for top-level fields (credit_score, monthly_payments, credit_freeze, fraud_alert, deceased, inquiries)
    try:
        doc_rec = extract_pdf_all_fields(str(pdf_path), include_spans=True)
        lines = doc_rec.get('all_lines_obj') or doc_rec.get('lines') or []

        def _line_text(ln):
            # safe join
            return ''.join([s.get('text','') for s in ln.get('spans', [])]).strip() if ln.get('spans') else ln.get('text','').strip() if ln.get('text') else ''

        def _find_by_text(t):
            if not t:
                return None
            for ln in lines:
                txt = _line_text(ln)
                if txt == t or t in txt or txt in t:
                    return ln
            return None

        # credit score (match numeric string)
        if gt.get('credit_score') is not None:
            cs_txt = str(gt.get('credit_score'))
            ln = _find_by_text(cs_txt)
            if ln:
                gt['credit_score_bbox'] = ln.get('bbox')
                gt['credit_score_page'] = ln.get('page')
                gt['credit_score_spans'] = ln.get('spans')
        # monthly payments (try $amount, raw amount, or '/mo' lines)
        if gt.get('monthly_payments') is not None:
            amt = gt.get('monthly_payments')
            candidates = [f'${amt}', str(amt)]
            found = None
            for c in candidates:
                found = _find_by_text(c)
                if found:
                    break
            if not found:
                for ln in lines:
                    txt = _line_text(ln)
                    if '/mo' in txt.lower() or '/month' in txt.lower():
                        found = ln
                        break
            if found:
                gt['monthly_payments_bbox'] = found.get('bbox')
                gt['monthly_payments_page'] = found.get('page')
                gt['monthly_payments_spans'] = found.get('spans')

        # credit_freeze, fraud_alert, deceased: find heading line and take next line's spans
        for key, heading in (('credit_freeze','credit freeze'), ('fraud_alert','fraud alert'), ('deceased','deceased')):
            if key in gt:
                for idx, ln in enumerate(lines):
                    txt = _line_text(ln).lower()
                    if heading in txt and idx+1 < len(lines):
                        nxt = lines[idx+1]
                        gt[f'{key}_bbox'] = nxt.get('bbox')
                        gt[f'{key}_page'] = nxt.get('page')
                        gt[f'{key}_spans'] = nxt.get('spans')
                        break

        # inquiries: match numeric count next to 'inquires' text
        if gt.get('inquiries_last_6_months') is not None:
            val = str(gt.get('inquiries_last_6_months'))
            # prefer a numeric match whose prior line mentions 'inquiries'
            for idx, ln in enumerate(lines):
                txt = _line_text(ln)
                if val == txt or val in txt:
                    prev = lines[idx-1] if idx > 0 else None
                    prev_txt = _line_text(prev).lower() if prev else ''
                    if 'inquir' in prev_txt or 'inquir' in txt.lower():
                        gt['inquiries_last_6_months_bbox'] = ln.get('bbox')
                        gt['inquiries_last_6_months_page'] = ln.get('page')
                        gt['inquiries_last_6_months_spans'] = ln.get('spans')
                        break
    except Exception:
        # best-effort: do not raise on mapping failures
        pass
    # write enriched GT to a new file
    enriched = gt_json_path.with_name(gt_json_path.stem + '.with_spans.json')
    enriched.write_text(json.dumps(gt, indent=2), encoding='utf-8')
    tmp_unvalidated.unlink(missing_ok=True)
    return enriched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('pdfs', nargs='+')
    ap.add_argument('--include-spans', action='store_true')
    ap.add_argument('--out', help='override output path')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--backup', action='store_true')
    args = ap.parse_args(argv)

    for p in args.pdfs:
        pdf = Path(p)
        if not pdf.exists():
            print('missing PDF', p); continue
        rec = extract_pdf_all_fields(str(pdf), include_spans=args.include_spans)
        gt = build_text_only_gt(rec, include_spans=args.include_spans)
        outp = Path(args.out) if args.out else default_out_path(pdf)
        if args.dry_run:
            print('DRY-RUN would write:', outp)
            print(json.dumps(gt, indent=2)[:1000])
            if args.include_spans:
                # For convenience, run the auto-mapper on a temporary copy so the user can preview attached spans
                import tempfile
                tmp = Path(tempfile.mkdtemp()) / (outp.name + '.dryrun.json')
                tmp.write_text(json.dumps(gt, indent=2), encoding='utf-8')
                try:
                    enriched = attach_spans_to_gt(tmp, pdf)
                    print('\nDRY-RUN enriched preview (first 1000 chars):')
                    print(enriched.read_text(encoding='utf-8')[:1000])
                    print('\nDRY-RUN spans were attached (preview only, not written)')
                except Exception as e:
                    print('DRY-RUN span-attachment failed:', e)
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                        tmp.parent.rmdir()
                    except Exception:
                        pass
            continue
        # backup if target exists
        if outp.exists() and args.backup:
            bak = outp.with_suffix(outp.suffix + '.bak')
            bak.write_text(outp.read_text(encoding='utf-8'), encoding='utf-8')
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(gt, indent=2), encoding='utf-8')
        print('WROTE', outp)
        if args.include_spans:
            enriched = attach_spans_to_gt(outp, pdf)
            print('WROTE enriched spans JSON', enriched)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())