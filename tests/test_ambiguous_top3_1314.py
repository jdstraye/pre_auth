import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_top3_json_exists():
    p = ROOT / 'data' / 'poc_ambiguous_diagnostics_user_1314_top3.json'
    assert p.exists(), f"Top3 diagnostics JSON missing: {p}"


def test_three_candidates_each():
    p = ROOT / 'data' / 'poc_ambiguous_diagnostics_user_1314_top3.json'
    data = json.loads(p.read_text())
    assert len(data) == 12, f'Expected 12 phrases, got {len(data)}'
    for rec in data:
        assert 'candidates' in rec and len(rec['candidates']) >= 3, f"Phrase {rec.get('phrase')} has <3 candidates"


def test_overlays_exist_and_matched_spans():
    p = ROOT / 'data' / 'poc_ambiguous_diagnostics_user_1314_top3.json'
    data = json.loads(p.read_text())
    for rec in data:
        top1 = rec['candidates'][0]
        ov = Path(top1['overlay'])
        assert ov.exists(), f"Overlay missing: {ov}"
        # at least one matched span across top3
        any_match = any(len(c.get('matched_spans',[]))>0 for c in rec['candidates'])
        assert any_match, f"No matched spans for phrase {rec.get('phrase')}"


def test_csv_pointing_to_overlay():
    csvp = ROOT / 'data' / 'poc_qa_ambiguous.csv'
    with open(csvp) as fh:
        r = csv.DictReader(fh)
        for row in r:
            if row['uncertain'].strip().lower() in ('true','1','yes'):
                img = row.get('sample_img','')
                assert img, f"Uncertain row missing sample_img for {row['phrase']}"
                assert Path(img).exists(), f"sample_img path does not exist: {img}"
