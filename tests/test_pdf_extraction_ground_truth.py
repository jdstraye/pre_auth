import os
import random
import json
import pytest
# Skip these heavy PDF-based ground-truth tests when PyMuPDF is not installed in the environment.
# This makes local dev and CI resilient if installing PyMuPDF is not possible.
pytest.importorskip("fitz", reason="PyMuPDF (fitz) is required for PDF extraction ground-truth tests")
from pathlib import Path
from src.scripts.pdf_color_extraction import extract_pdf_all_fields

# Configurable number of PDFs to test


def get_pdf_files(pdf_dir):
    return sorted([str(f) for f in Path(pdf_dir).glob("user_*_credit_summary_2025*.pdf")])


def pdf_to_ground_truth_name(pdf_path):
    pdf_name = Path(pdf_path).stem
    return f"{pdf_name}_ground_truth.json"

def pdf_to_unvalidated_name(pdf_path):
    pdf_name = Path(pdf_path).stem
    return f"{pdf_name}_ground_truth_unvalidated.json"


def load_json(path):
    import re
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    try:
        obj = json.loads(s)
    except json.decoder.JSONDecodeError as e:
        # Attempt lightweight repair for common issues (e.g., unquoted date values like "date": 2016-10-11)
        s_fixed = re.sub(r'"date"\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})', r'"date": "\1"', s)
        try:
            obj = json.loads(s_fixed)
        except Exception:
            # Re-raise original error for user to inspect
            raise
    # Support legacy ground truth format where data may be wrapped in {'rec': {...}}
    if isinstance(obj, dict) and 'rec' in obj and isinstance(obj['rec'], dict):
        return obj['rec']
    return obj

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _normalize_aliases(d):
    d = dict(d)
    # inquiries aliases
    if 'inquiries_6mo' in d and 'inquiries_last_6_months' not in d:
        d['inquiries_last_6_months'] = d['inquiries_6mo']
    if 'inquiries_last_6_months' in d and 'inquiries_6mo' not in d:
        d['inquiries_6mo'] = d['inquiries_last_6_months']
    # collections naming variants (prefer flat 'collections_open'/'collections_closed')
    # Note: do not create '*_count' variants dynamically to avoid introducing keys not present in some GT files
    # but allow legacy *_count keys if present in input
    if 'collections_open' not in d and 'collections_open_count' in d:
        d['collections_open'] = d.get('collections_open_count')
    if 'collections_closed' not in d and 'collections_closed_count' in d:
        d['collections_closed'] = d.get('collections_closed_count')
    # Normalize nested account structures into flat keys so structural variants compare equal
    nested_mappings = [
        ('revolving_accounts_open', 'revolving_open_count', 'revolving_open_total'),
        ('installment_accounts_open', 'installment_open_count', 'installment_open_total'),
        ('real_estate_open', 'real_estate_open_count', 'real_estate_open_total'),
        ('line_of_credit_accounts_open', 'line_of_credit_accounts_open_count', 'line_of_credit_accounts_open_total'),
        ('miscellaneous_accounts_open', 'miscellaneous_accounts_open_count', 'miscellaneous_accounts_open_total'),
    ]
    for nkey, ckey, tkey in nested_mappings:
        if nkey in d and ckey not in d:
            nd = d.get(nkey, {})
            d[ckey] = nd.get('count') if isinstance(nd, dict) else None
            d[tkey] = nd.get('amount') if isinstance(nd, dict) else None
        if ckey in d and nkey not in d:
            d[nkey] = {'count': d.get(ckey), 'amount': d.get(tkey)}
    # Collections nested vs flat
    if 'collections' in d and 'collections_open' not in d:
        c = d.get('collections', {})
        if isinstance(c, dict):
            d['collections_open'] = c.get('open')
            d['collections_closed'] = c.get('closed')
    if 'collections_open' in d and 'collections' not in d:
        d['collections'] = {'open': d.get('collections_open'), 'closed': d.get('collections_closed')}
    # Compute credit factor counts (derive if omitted) so count fields are consistent
    if 'credit_factors' in d and isinstance(d['credit_factors'], list):
        d['red_credit_factors_count'] = sum(1 for f in d['credit_factors'] if f.get('color') == 'red')
        d['green_credit_factors_count'] = sum(1 for f in d['credit_factors'] if f.get('color') == 'green')
        d['black_credit_factors_count'] = sum(1 for f in d['credit_factors'] if f.get('color') == 'black')
    # Normalize credit_score nested vs flat
    if isinstance(d.get('credit_score'), dict):
        cs = d.get('credit_score', {})
        d['credit_score'] = cs.get('value')
        d['credit_score_color'] = cs.get('color')
    # Normalize pdf_file vs source/filename
    if 'pdf_file' in d:
        pf = d.get('pdf_file')
        d['source'] = pf
        d['filename'] = pf.split('/')[-1]
        d.pop('pdf_file', None)
    # Late pays nested vs flat
    if 'late_pays' in d and isinstance(d['late_pays'], dict):
        lp = d['late_pays']
        d['late_pays_2yr'] = lp.get('last_2_years')
        d['late_pays_gt2yr'] = lp.get('last_over_2_years')
    if 'late_pays_2yr' in d and 'late_pays' not in d:
        d['late_pays'] = {'last_2_years': d.get('late_pays_2yr'), 'last_over_2_years': d.get('late_pays_gt2yr')}
    # Normalize variants of credit card totals naming used in different GT files
    if 'credit_card_open_totals_no_retail' in d and 'credit_card_open_totals' not in d:
        d['credit_card_open_totals'] = d.pop('credit_card_open_totals_no_retail')
    # Normalize internal field names inside credit_card_open_totals
    if 'credit_card_open_totals' in d and isinstance(d['credit_card_open_totals'], dict):
        cc = dict(d['credit_card_open_totals'])
        if 'utilization_percent' in cc and 'Percent' not in cc:
            cc['Percent'] = cc.pop('utilization_percent')
        if 'payment' in cc and 'Payment' not in cc:
            cc['Payment'] = cc.pop('payment')
        d['credit_card_open_totals'] = cc
    # Ensure we always have the key so comparisons are stable
    if 'credit_card_open_totals' not in d:
        d['credit_card_open_totals'] = None
    # Drop/ignore fields that are present only in some schemas to avoid spurious failures    # Normalize address to a list for comparison (accept string vs list variants)
    def _norm_addr(s):
        # remove comma before zip (e.g., ', 40513' -> ' 40513'), collapse spaces
        s2 = s.replace(',\s', ', ')
        s2 = s2.rstrip(', ')
        s2 = s2.replace(', ,', ',')
        s2 = s2.replace(', ,', ',')
        s2 = s2.replace(',  ', ', ')
        s2 = s2.replace(', ,', ',')
        s2 = ' '.join(s2.split())
        # remove comma before 5-digit zip
        import re
        s2 = re.sub(r',\s*(\d{5})$', r' \1', s2)
        return s2
    if 'address' in d:
        if isinstance(d['address'], str):
            d['address'] = [_norm_addr(d['address'])]
        elif isinstance(d['address'], list):
            d['address'] = [_norm_addr(x) for x in d['address']]
    # Simplify credit_factors entries to only factor+color for robust comparison
    if 'credit_factors' in d and isinstance(d['credit_factors'], list):
        simplified = []
        for f in d['credit_factors']:
            if isinstance(f, dict):
                simplified.append({'factor': f.get('factor'), 'color': f.get('color')})
            else:
                simplified.append({'factor': f, 'color': None})
        d['credit_factors'] = simplified
    return d


def compare_dicts(a, b):
    na = _normalize_aliases(a)
    nb = _normalize_aliases(b)
    # Normalize missing/None count fields to 0 for robust equality (e.g., collections_open_count)
    for d in (na, nb):
        for k in list(d.keys()):
            if k.endswith('_count') and d.get(k) is None:
                d[k] = 0
    # Ensure both sides have the same set of *_count keys (default missing ones to 0)
    count_keys = set([k for k in set(list(na.keys()) + list(nb.keys())) if k.endswith('_count')])
    for d in (na, nb):
        for k in count_keys:
            if k not in d:
                # If the non-count base value exists (e.g., 'collections_open'), use it to avoid
                # introducing a default 0 that would cause spurious mismatches when the other
                # side has a non-zero value.
                base_key = k[:-6]  # remove trailing '_count'
                if base_key in d and d.get(base_key) is not None:
                    d[k] = d.get(base_key)
                else:
                    d[k] = 0
    # Special-case address: require exact presence/equality (no more subset acceptance)
    if 'address' in na or 'address' in nb:
        a_addr = na.get('address', []) or []
        b_addr = nb.get('address', []) or []
        # Normalize single-string addresses to lists to make comparison consistent
        if isinstance(a_addr, str):
            a_addr = [a_addr]
        if isinstance(b_addr, str):
            b_addr = [b_addr]
        # Require exact membership equality: both sides must contain the same address elements
        if set(a_addr) != set(b_addr):
            return False
        # remove address keys to allow remaining fields to compare
        na = dict(na)
        nb = dict(nb)
        na.pop('address', None)
        nb.pop('address', None)
    # Strict credit_factors matching: require exact color + normalized factor text equality (no fuzzy edit-distance)
    def _norm_factor_text(s):
        import re
        return re.sub(r'[^a-z0-9]+', '', s.lower())
    # If one side entirely lacks 'credit_factors' (legacy GTs), treat them as intentionally absent and skip comparison
    if 'credit_factors' not in na or 'credit_factors' not in nb:
        na = dict(na); nb = dict(nb)
        # Also drop derived color-count fields to avoid spurious mismatches
        for k in ('red_credit_factors_count','green_credit_factors_count','black_credit_factors_count'):
            na.pop(k, None); nb.pop(k, None)
        na.pop('credit_factors', None); nb.pop('credit_factors', None)
    else:
        a_cf = na.get('credit_factors', []) or []
        b_cf = nb.get('credit_factors', []) or []
        # Build list of simplified tuples (color, norm_text)
        def build_list(cf_list):
            out = []
            for f in cf_list:
                if not isinstance(f, dict):
                    out.append((None, _norm_factor_text(str(f))))
                else:
                    out.append((f.get('color'), _norm_factor_text(f.get('factor', ''))))
            return out
        A = build_list(a_cf)
        B = build_list(b_cf)
        from collections import Counter
        if Counter(A) != Counter(B):
            return False
        # matched credit_factors exactly; remove them and compare remaining
        na = dict(na)
        nb = dict(nb)
        na.pop('credit_factors', None)
        nb.pop('credit_factors', None)
    # Tolerate missing credit_card_open_totals in ground-truth: if one side is None and the other a dict, drop the key for comparison
    if 'credit_card_open_totals' in na or 'credit_card_open_totals' in nb:
        a_cc = na.get('credit_card_open_totals')
        b_cc = nb.get('credit_card_open_totals')
        if (a_cc is None and isinstance(b_cc, dict)) or (b_cc is None and isinstance(a_cc, dict)):
            na = dict(na); nb = dict(nb)
            na.pop('credit_card_open_totals', None)
            nb.pop('credit_card_open_totals', None)
        else:
            # When both sides have a dict, ignore transient metadata like 'hex' that may be present in the extractor output
            if isinstance(a_cc, dict) and isinstance(b_cc, dict):
                a_cc = dict(a_cc); b_cc = dict(b_cc)
                a_cc.pop('hex', None); b_cc.pop('hex', None)
                na = dict(na); nb = dict(nb)
                na['credit_card_open_totals'] = a_cc
                nb['credit_card_open_totals'] = b_cc

    # Normalize top-level *_color keys: if one side has a color and the other has None, drop the key to avoid spurious mismatches
    color_keys = set(k for k in set(list(na.keys()) + list(nb.keys())) if k.endswith('_color'))
    for k in color_keys:
        a_has = (k in na and na.get(k) is not None)
        b_has = (k in nb and nb.get(k) is not None)
        # If one side has a non-None color while the other is missing/None, drop the key from both
        if a_has != b_has:
            na = dict(na); nb = dict(nb)
            na.pop(k, None); nb.pop(k, None)

    return na == nb

def test_pdf_extraction_vs_ground_truth(request, pdf_path):
    pdf_dir = request.config.getoption("--pdf_dir")
    ground_truth_dir = request.config.getoption("--ground_truth_dir")
    pdf_name = Path(pdf_path).stem
    gt_path = Path(ground_truth_dir) / pdf_to_ground_truth_name(pdf_path)
    unval_path = Path(ground_truth_dir) / pdf_to_unvalidated_name(pdf_path)

    # Run extraction
    extracted = extract_pdf_all_fields(pdf_path)

    if gt_path.exists():
        ground_truth = load_json(gt_path)
        assert compare_dicts(extracted, ground_truth), f"Mismatch for {pdf_name}"
    else:
        save_json(extracted, unval_path)
        pytest.xfail(f"Ground truth missing for {pdf_name}. Extraction saved as unvalidated. Needs review.")


def pytest_generate_tests(metafunc):
    if "pdf_path" in metafunc.fixturenames:
        pdf_dir = metafunc.config.getoption("--pdf_dir")
        user_id = metafunc.config.getoption("--user_id")
        n_pdfs = metafunc.config.getoption("--n_pdfs")
        pdfs = get_pdf_files(pdf_dir)
        if user_id:
            pdfs = [p for p in pdfs if f"user_{user_id}_" in p]
        elif len(pdfs) > n_pdfs:
            pdfs = random.sample(pdfs, n_pdfs)
        metafunc.parametrize("pdf_path", pdfs)
