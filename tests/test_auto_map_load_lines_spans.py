from scripts.auto_map_unvalidated import load_doc_lines
import json


def test_load_doc_lines_requests_spans(monkeypatch, tmp_path):
    # monkeypatch the extractor to assert include_spans=True is honored
    called = {}

    def fake_extract(path, include_spans=False):
        called['include_spans'] = include_spans
        return {'all_lines_obj': [{'page': 0, 'spans':[{'text':'x','hex':'#fff','bbox':[1,2,3,4]}], 'bbox':[1,2,3,4]}]}

    monkeypatch.setattr('scripts.auto_map_unvalidated.extract_pdf_all_fields', fake_extract)
    lines = load_doc_lines('data/pdf_analysis/foo.pdf')
    assert isinstance(lines, list)
    assert called.get('include_spans') is True
    assert lines[0]['spans'][0]['text'] == 'x'
