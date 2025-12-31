#!/usr/bin/env python3
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.poc_extract_credit_factors import extract_credit_factors_from_doc

DATA_DIR = ROOT / 'data'
PDF_DIR = DATA_DIR / 'pdf_analysis'
COLOR_DIR = DATA_DIR / 'color_training'
OUT_DIR = DATA_DIR / 'agent_validation'
OUT_DIR.mkdir(exist_ok=True)


def load_agent_labels(pdf_path):
    # find matching color_training phrases file
    base = Path(pdf_path).stem  # user_150_credit_summary_...
    # match .p1.right.phrases.txt
    ph_file = COLOR_DIR / f"{base}.p1.right.phrases.txt"
    if not ph_file.exists():
        return None
    out = []
    for line in ph_file.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            out.append({'factor': obj.get('phrase'), 'color': obj.get('cat')})
        except Exception:
            continue
    return out


def extract_script_labels(pdf_path):
    import fitz
    doc = fitz.open(str(pdf_path))
    factors = extract_credit_factors_from_doc(doc)
    # factors is list of dicts with 'factor' and 'color'
    return [{'factor': f.get('factor'), 'color': f.get('color')} for f in factors]


def compare_lists(script, agent):
    # return mismatches as dict
    s_map = {s['factor'].strip().lower(): s['color'] for s in script}
    a_map = {a['factor'].strip().lower(): a['color'] for a in agent}
    missing_in_script = [f for f in a_map if f not in s_map]
    missing_in_agent = [f for f in s_map if f not in a_map]
    color_mismatches = []
    for f, acol in a_map.items():
        scol = s_map.get(f)
        if scol and scol != acol:
            color_mismatches.append({'factor': f, 'script': scol, 'agent': acol})
    return {'missing_in_script': missing_in_script, 'missing_in_agent': missing_in_agent, 'color_mismatches': color_mismatches}


if __name__=='__main__':
    if len(sys.argv) < 2:
        print('usage: validate_credit_factors.py <pdf_path>')
        sys.exit(2)
    pdf_path = Path(sys.argv[1])
    print('Validating', pdf_path)
    script = extract_script_labels(pdf_path)
    agent = load_agent_labels(pdf_path)
    out_script = OUT_DIR / f'script_{pdf_path.stem}.json'
    out_agent = OUT_DIR / f'agent_{pdf_path.stem}.json'
    out_script.write_text(json.dumps(script, indent=2))
    out_agent.write_text(json.dumps(agent, indent=2))
    print('Wrote', out_script, out_agent)
    comp = compare_lists(script, agent if agent is not None else [])
    print(json.dumps(comp, indent=2))
    # return nonzero when mismatches exist
    if comp['missing_in_script'] or comp['missing_in_agent'] or comp['color_mismatches']:
        sys.exit(1)
    print('MATCH')
    sys.exit(0)
