from scripts.poc_extract_credit_factors import parse_public_records


def test_parse_public_records_with_number():
    txt = "Some header\nPublic Records: 0\nOther stuff"
    c, n = parse_public_records(txt)
    assert c == 0
    assert n == ''


def test_parse_public_records_with_number_on_newline():
    txt = "Credit Alerts\nPublic Records\n1\nMore"
    c, n = parse_public_records(txt)
    assert c == 1
    assert n == ''


def test_parse_public_records_with_bankruptcy_note():
    txt = "Credit Alerts\nPublic Records\n\nBankruptcy Discharged CH-7Discharged CH-7 - 10/13/2020"
    c, n = parse_public_records(txt)
    assert c == 1
    assert 'bankrupt' in n.lower() or 'bankruptcy' in n.lower() or 'bk' in n.lower()