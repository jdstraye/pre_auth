from scripts.poc_extract_credit_factors import normalize_factors


def test_normalize_sample_raw_factors():
    raw = [
        {'factor': 'Revolving Accounts (Open)', 'color':'black','hex':'#202429'},
        {'factor': '2 / $1,104', 'color':'green','hex':'#ebeff4'},
        {'factor': 'Line of Credit Accounts (Open)', 'color':'black','hex':'#202429'},
        {'factor': '1 / $466', 'color':'red','hex':'#ac2f2d'},
        {'factor': 'Installment Accounts (Open)', 'color':'black','hex':'#202429'},
        {'factor': '3 / $26,491', 'color':'green','hex':'#ebeff4'},
        {'factor': '$466', 'color':'red','hex':'#ac2f2d'},
    ]
    norm = normalize_factors(raw)
    # Expect canonical entries for revolving, line of credit, and installment with counts/totals
    keys = {f['canonical'] for f in norm}
    assert 'revolving_accounts_open' in keys or any('revolving' in k for k in keys)
    assert any(f.get('count') == 2 and f.get('total') == 1104 for f in norm)
    assert any(f.get('count') == 3 and f.get('total') == 26491 for f in norm)
    # standalone $466 should be removed or absorbed
    assert not any(f.get('label') == '$466' for f in norm)
