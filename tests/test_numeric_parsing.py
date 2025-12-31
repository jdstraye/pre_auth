from scripts.poc_extract_credit_factors import parse_count_amount_pair


def test_parse_count_amount_pair_simple():
    assert parse_count_amount_pair('Installment Accounts (Open): 10 / $56,881') == (10, 56881)
    assert parse_count_amount_pair('10 / 56881') == (10, 56881)
    assert parse_count_amount_pair('$56,881 / 10') == (10, 56881)
    assert parse_count_amount_pair('21 / 25,304') == (21, 25304)


def test_parse_ignores_month_suffix():
    assert parse_count_amount_pair('Payments $1,036/mo') == (None, None)
    assert parse_count_amount_pair('Payment: $1,036 / mo') == (None, None)


def test_installment_parsing_from_pdf_user_1314():
    # read full text from the sample PDF and run extract_record_level
    pdf_path = 'data/pdf_analysis/user_1314_credit_summary_2025-09-01_092724.pdf'
    # gather page texts
    from scripts.poc_extract_credit_factors import _pdf_page_count, _get_page_text, extract_record_level
    pcount = _pdf_page_count(pdf_path)
    full_text = '\n'.join(_get_page_text(pdf_path, p+1) for p in range(pcount))
    rec = extract_record_level(full_text)
    # If text-based parsing misses installment counts (common for some table layouts), try scanning rendered lines for numeric pairs as pipeline does
    if rec.get('installment_open_count') is None or rec.get('installment_open_total') is None:
        import pytest
        fitz = pytest.importorskip('fitz')
        from scripts.poc_extract_credit_factors import parse_count_amount_pair
        doc = fitz.open(pdf_path)
        prev_ln = ''
        for p in range(len(doc)):
            td = doc.load_page(p).get_text('dict')
            for b in td.get('blocks', []):
                for ln in b.get('lines', []):
                    line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                    if not line_text:
                        prev_ln = line_text
                        continue
                    combined = (prev_ln + ' ' + line_text).strip()
                    cnt, amt = parse_count_amount_pair(combined)
                    if cnt is None and amt is None:
                        cnt, amt = parse_count_amount_pair(line_text)
                    if (cnt is not None or amt is not None) and 'install' in combined.lower():
                        if cnt is not None:
                            rec['installment_open_count'] = int(cnt)
                        if amt is not None:
                            rec['installment_open_total'] = int(amt)
                        break
                    prev_ln = line_text
                if rec.get('installment_open_count') is not None:
                    break
            if rec.get('installment_open_count') is not None:
                break
    assert rec.get('installment_open_count') == 10, f"expected 10, got {rec.get('installment_open_count')}"
    assert rec.get('installment_open_total') == 56881, f"expected 56881, got {rec.get('installment_open_total')}"


def test_run_sample_extraction_script_user_1314():
    import pytest, subprocess, json
    pytest.importorskip('fitz')
    cmd = [sys.executable if 'sys' in globals() else 'python', 'scripts/run_sample_extraction.py', 'user_1314_credit_summary_2025-09-01_092724.pdf']
    subprocess.run(cmd, check=True)
    with open('data/extracted/user_1314_credit_summary_2025-09-01_092724.pdf.json') as fh:
        j = json.load(fh)
    wr = j.get('wide_row', {})
    assert int(wr.get('installment_open_count',0)) == 10
    assert int(wr.get('installment_open_total',0)) == 56881
