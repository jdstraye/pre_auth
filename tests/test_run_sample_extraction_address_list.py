import json
import subprocess
from pathlib import Path

ROOT = Path('.').resolve()
PDF_DIR = ROOT / 'data' / 'pdf_analysis'
OUT = ROOT / 'data' / 'extracted'


def test_run_sample_extraction_handles_list_addresses():
    pdf_name = 'user_1246_credit_summary_2025-09-01_105842.pdf'
    cmd = ['python3', 'scripts/run_sample_extraction.py', pdf_name]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"script failed: stdout={res.stdout} stderr={res.stderr}"
    outp = OUT / f"{pdf_name}.json"
    assert outp.exists(), f"Expected output file {outp} to exist"
    data = json.loads(outp.read_text())
    rec = data.get('rec')
    assert rec is not None
    assert 'address' in rec
    # address may be a string or a list; ensure it's non-empty
    addr = rec['address']
    if isinstance(addr, list):
        assert len(addr) >= 1
        assert all(a and a.strip() for a in addr if isinstance(a, str))
    else:
        assert isinstance(addr, str) and addr.strip()
