#!/usr/bin/env python3
import json, re, os, csv, sys, subprocess
from pathlib import Path
import boto3, requests

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "prefi_weaviate_clean-2.json"
OUT = ROOT / "data" / "pdf_scan.csv"
TMP_DIR = ROOT / "data" / "pdf_samples"
S3_PREFIX = re.compile(r"^s3://([^/]+)/(.+)$", re.I)


def is_pdf_url(s):
    return isinstance(s, str) and (s.lower().endswith('.pdf') or '.pdf' in s.lower())


def find_pdf_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from find_pdf_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_pdf_values(v)
    elif isinstance(obj, str):
        if is_pdf_url(obj):
            yield obj


def head_s3(uri):
    m = S3_PREFIX.match(uri)
    if not m:
        return None
    bucket, key = m.group(1), m.group(2)
    s3 = boto3.client("s3")
    try:
        r = s3.head_object(Bucket=bucket, Key=key)
        return {"ok": True, "size": r.get("ContentLength"), "last_modified": r.get("LastModified").isoformat()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def head_http(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return {"ok": r.status_code < 400, "status_code": r.status_code, "content_length": r.headers.get("content-length")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def pdftotext_sample(local_path):
    try:
        out = subprocess.check_output(["pdftotext", "-l", "1", str(local_path), "-"], stderr=subprocess.DEVNULL, timeout=20)
        return len(out.strip()), out.decode('utf-8', errors='replace')[:400]
    except Exception:
        return 0, ""


def download_s3(uri, dest):
    m = S3_PREFIX.match(uri)
    bucket, key = m.group(1), m.group(2)
    boto3.client("s3").download_file(bucket, key, str(dest))


def download_http(url, dest):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main():
    seen = set()
    rows = []
    if not IN.exists():
        print("ERROR: input json not found:", IN, file=sys.stderr)
        sys.exit(2)
    data = json.loads(IN.read_text())
    for val in find_pdf_values(data):
        if val in seen: continue
        seen.add(val)
        row = {"source": val}
        try:
            if val.startswith("s3://"):
                row.update(head_s3(val) or {})
            elif val.lower().startswith("http"):
                row.update(head_http(val) or {})
            else:
                row.update({"ok": False, "error": "unknown-scheme"})
        except Exception as e:
            row.update({"ok": False, "error": str(e)})
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source","ok","size","status_code","last_modified","error"])
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in ["source","ok","size","status_code","last_modified","error"]})
    print("Wrote", OUT)

    # pick up to 3 candidate samples (ok==True)
    samples = [r["source"] for r in rows if r.get("ok")]
    chosen = samples[:3]
    for i, s in enumerate(chosen, start=1):
        dest = TMP_DIR / f"sample_{i}.pdf"
        try:
            if s.startswith("s3://"):
                print("Downloading", s)
                download_s3(s, dest)
            else:
                print("Downloading", s)
                download_http(s, dest)
            nchars, text = pdftotext_sample(dest)
            print(f"Sample {i}: {s} -> pdftotext chars={nchars}")
            # write text sample for quick inspection
            (TMP_DIR / f"sample_{i}.txt").write_text(text)
        except Exception as e:
            print("Sample download failed", s, e)

    print("Done")


if __name__ == "__main__":
    main()
