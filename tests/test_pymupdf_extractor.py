import pytest
from pathlib import Path
import os

# skip if fitz not available
pytest.importorskip('fitz')
import fitz

from scripts.poc_extract_credit_factors import (
    load_expectations_from_dir,
    find_credit_factors_region,
    extract_lines_from_region,
    span_color_hex,
    map_color_to_cat,
    combined_sample_color_for_phrase,
    ROOT,
)

PDF_DIR = ROOT / 'data' / 'pdf_analysis'


def normalize_phrase(s: str) -> str:
    return ' '.join(''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in s).split())


def test_pymupdf_extracts_expected_colors_for_1314():
    # Load expectations
    ex = load_expectations_from_dir(ROOT / 'data' / 'pdf_analysis')
    fname = None
    for f in ex:
        if '1314' in f:
            fname = f
            break
    assert fname, "Expectation file for 1314 not found"

    expects = ex[fname]
    pdf_path = PDF_DIR / fname
    if not pdf_path.exists():
        # fallback to PDF_DIR global poc dir
        pdf_path = ROOT / 'data' / 'poc_pdfs' / fname
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"

    doc = fitz.open(str(pdf_path))
    # Use span-only phrase sampling via combined_sample_color_for_phrase for each expectation
    found_map = {}
    for phrase, expected_color in expects.items():
        try:
            res = combined_sample_color_for_phrase(doc, phrase, page_limit=3)
            if res:
                pidx, text, hexv, rgb, bbox, method = res
                found_map[normalize_phrase(text)] = {'page': pidx, 'text': text, 'hex': hexv, 'rgb': rgb, 'cat': map_color_to_cat(rgb), 'bbox': bbox, 'method': method}
        except Exception:
            pass
    # Span-only collection: scan all pages/lines and use span color information directly
    found_map = {}
    for p in range(len(doc)):
        td = doc.load_page(p).get_text('dict')
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if not text:
                    continue
                norm = normalize_phrase(text)
                hexv, rgb = span_color_hex(ln.get('spans', []))
                cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
                existing = found_map.get(norm)
                if existing is None or (existing.get('cat','neutral') == 'neutral' and cat != 'neutral'):
                    found_map[norm] = {'page': p, 'text': text, 'hex': hexv, 'rgb': rgb, 'cat': cat, 'bbox': ln.get('bbox')}
    # Validate each expectation against the span-only sampling results we collected
    def useful_tokens(s):
        toks = [t for t in s.split() if len(t) > 2 and not t.isdigit()]
        return set(toks)

    errors = []
    for phrase, expected_color in expects.items():
        norm_ph = normalize_phrase(phrase)
        # direct hit
        found = found_map.get(norm_ph)
        if not found:
            # attempt token-overlap best match with tie-breakers
            target_tokens = useful_tokens(norm_ph)
            candidates = []
            best_score = 0
            for k, v in found_map.items():
                score = len(target_tokens & useful_tokens(k))
                if score > 0:
                    candidates.append((score, k, v))
                    if score > best_score:
                        best_score = score
            min_ok = max(1, int(max(1, len(target_tokens) * 0.4)))
            if best_score < min_ok:
                errors.append(f"Phrase not found: '{phrase}' (norm: '{norm_ph}')")
                continue
            top = [(k, v) for s, k, v in candidates if s == best_score]
            found = None
            if len(top) == 1:
                found = top[0][1]
            else:
                # tie-breaker 1: numeric token match
                nums = [t for t in norm_ph.split() if any(ch.isdigit() for ch in t)]
                if nums:
                    for k, v in top:
                        ktoks = k.split()
                        if any(n in ktoks for n in nums):
                            found = v
                            break
                if found is None:
                    # tie-breaker 2: prefer longer unique token overlap
                    best_u = None; best_u_score = -1
                    for k, v in top:
                        uniq_score = sum(len(t) for t in target_tokens if len(t) > 4 and t in k.split())
                        if uniq_score > best_u_score:
                            best_u_score = uniq_score; best_u = v
                    if best_u_score > 0:
                        found = best_u
                if found is None:
                    # tie-breaker 3: bigram overlap
                    def bigrams(toklist):
                        return set(' '.join(toklist[i:i+2]) for i in range(max(0,len(toklist)-1)))
                    tg = bigrams(norm_ph.split())
                    best_b = None; best_b_score = -1
                    for k, v in top:
                        kg = bigrams(k.split())
                        bscore = len(tg & kg)
                        if bscore > best_b_score:
                            best_b_score = bscore; best_b = v
                    if best_b is not None:
                        found = best_b
                if found is None:
                    # fallback: pick first
                    found = top[0][1]
        sampled_cat = found.get('cat', 'neutral')
        if sampled_cat != expected_color:
            errors.append(f"Color mismatch for '{phrase}': expected={expected_color} sampled={sampled_cat} page={found.get('page')}")
    if errors:
        raise AssertionError('\n'.join(errors))


def test_span_glyph_fallback_for_1314():
    # Focused test: prefer span color then glyph interior fallback for specific known failing lines
    pdf_path = PDF_DIR / 'user_1314_credit_summary_2025-09-01_092724.pdf'
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    doc = fitz.open(str(pdf_path))
    from scripts.poc_extract_credit_factors import color_first_search_for_phrase, map_color_to_cat

    res = color_first_search_for_phrase(doc, '6 Charged Off Accts', expected_color='green', page_limit=3)
    assert res is not None, 'Expected to find 6 Charged Off Accts with color'
    pidx, text, hx, rgb, page_bbox, pix_bbox, *rest = res
    assert map_color_to_cat(rgb) == 'green', f'Expected green for 6 Charged Off Accts, got {map_color_to_cat(rgb)}'

    res2 = color_first_search_for_phrase(doc, '8 Chrgd Off Rev Accts', expected_color='red', page_limit=3)
    assert res2 is not None, 'Expected to find 8 Chrgd Off Rev Accts with color'
    pidx2, text2, hx2, rgb2, pb2, pbx2, *rest2 = res2
    assert map_color_to_cat(rgb2) == 'red', f'Expected red for 8 Chrgd Off Rev Accts, got {map_color_to_cat(rgb2)}'


def test_span_detection_for_known_phrases():
    pdf_path = PDF_DIR / 'user_1314_credit_summary_2025-09-01_092724.pdf'
    assert pdf_path.exists()
    doc = fitz.open(str(pdf_path))
    # Check known phrases that previously relied on marker heuristics
    expect_map = {
        '6 Charged Off Accts': 'green',
        '8 Chrgd Off Rev Accts': 'red',
    }
    for phrase, exp in expect_map.items():
        res = combined_sample_color_for_phrase(doc, phrase, page_limit=3)
        assert res is not None, f"Expected to find phrase: {phrase}"
        _, text, hexv, rgb, bbox, method = res
        assert map_color_to_cat(rgb) == exp, f"Expected {exp} for '{phrase}', got {map_color_to_cat(rgb)}"


def test_prefer_best_candidate_for_ambiguous_phrases():
    # Ensure we pick the correct line when multiple lines share tokens using span-only sampling
    pdf_path = PDF_DIR / 'user_1314_credit_summary_2025-09-01_092724.pdf'
    assert pdf_path.exists()
    doc = fitz.open(str(pdf_path))

    for phrase in ['8+ Rev Accts with Balances', '1 Inq Last 4-5 mo']:
        res = combined_sample_color_for_phrase(doc, phrase, page_limit=3)
        assert res is not None, f'Expected to find phrase: {phrase}'
        _, text, hx, rgb, bbox, method = res
        cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
        assert cat == 'black' or cat == 'neutral', f"Expected black/neutral for '{phrase}', got {cat}"


def test_regression_1314_expectations_match():
    # Full QA regression: ensure the expectation-only QA reports zero mismatches for user_1314
    from scripts.poc_extract_credit_factors import run_expectation_only_qa, ROOT
    run_expectation_only_qa()
    import csv
    qa_sum = ROOT / 'data' / 'poc_qa_summary.csv'
    assert qa_sum.exists(), 'QA summary CSV not written'
    found = None
    with open(qa_sum, newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('filename','').startswith('user_1314'):
                found = row
                break
    assert found is not None, 'No QA summary row for user_1314'
    assert int(found.get('mismatches', 0)) == 0, f"Expected 0 mismatches for user_1314, got {found.get('mismatches')}"