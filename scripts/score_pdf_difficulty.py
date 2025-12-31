#!/usr/bin/env python3
import csv, subprocess, re, os
from pathlib import Path
import boto3
import fitz

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "pdf_scan.csv"
OUT = ROOT / "data" / "pdf_scores.csv"
PDF_DIR = ROOT / "data" / "pdf_analysis"
PDF_DIR.mkdir(parents=True, exist_ok=True)

S3_PREFIX = re.compile(r"^s3://([^/]+)/(.+)$", re.I)


def pdftotext_chars(pdf_path):
    try:
        out = subprocess.check_output(["pdftotext", "-l", "1", str(pdf_path), "-"], stderr=subprocess.DEVNULL, timeout=20)
        return len(out.strip())
    except Exception:
        return 0


def download_source(src, dest):
    if src.startswith("s3://"):
        m = S3_PREFIX.match(src)
        boto3.client("s3").download_file(m.group(1), m.group(2), str(dest))
    else:
        import requests
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)


def rgb_to_hex(rgb):
    try:
        r,g,b = [int(255*v) if isinstance(v,float) and v<=1 else int(v) for v in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return None


def analyze_pdf(pdf_path):
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    info = {"n_pages": len(doc)}
    textlen = pdftotext_chars(pdf_path)
    info["text_chars_first_page"] = textlen
    # count spans and distinct colors on page
    textdict = page.get_text("dict")
    spans = []
    for block in textdict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(span)
    info["n_spans"] = len(spans)
    colors = set()
    color_spans = 0
    for s in spans:
        col = s.get("color")
        if col:
            hexc = rgb_to_hex(col)
            if hexc and hexc != "#000000":
                colors.add(hexc)
                color_spans += 1
    info["n_color_spans"] = color_spans
    info["n_colors"] = len(colors)
    info["colors_sample"] = ";".join(list(colors)[:5])
    return info


def main():
    rows = []
    with open(IN_CSV) as fh:
        r = csv.DictReader(fh)
        for i,rec in enumerate(r, start=1):
            src = rec["source"]
            fname = src.split('/')[-1]
            dest = PDF_DIR / fname
            try:
                download_source(src, dest)
            except Exception as e:
                rows.append({"source": src, "ok": False, "error": str(e)})
                continue
            try:
                info = analyze_pdf(dest)
                recout = {"source": src, **info}
                rows.append(recout)
            except Exception as e:
                rows.append({"source": src, "ok": False, "error": str(e)})
    # score: higher score indicates harder
    scored = []
    for r in rows:
        if not r.get("n_spans"):
            score = 100 + (1 if r.get("n_color_spans") else 0)
        else:
            # base: low text chars -> harder
            chars = r.get("text_chars_first_page", 1000)
            score = max(0, 500 - chars)  # lower chars => higher score
            # add if multiple colors
            score += r.get("n_colors",0) * 50
            # longer docs slightly harder
            score += r.get("n_pages",1) * 5
        r["score"] = score
        scored.append(r)
    scored_sorted = sorted(scored, key=lambda x: x.get("score",0), reverse=True)
    # write out top 20
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source","n_pages","text_chars_first_page","n_spans","n_color_spans","n_colors","colors_sample","score","ok","error"])
        w.writeheader()
        for row in scored_sorted:
            w.writerow({k: row.get(k,"") for k in w.fieldnames})
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
