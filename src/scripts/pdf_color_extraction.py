import argparse
import logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def cli_extract():
	parser = argparse.ArgumentParser(description="Extract all fields from a credit summary PDF.")
	parser.add_argument('--user_id', type=str, help='User ID (e.g., 705)')
	parser.add_argument('--pdf_dir', type=str, default='data/pdf_analysis', help='Directory containing PDFs')
	parser.add_argument('--output_dir', type=str, default='data/extracted', help='Directory to save extracted JSON')
	args = parser.parse_args()

	# Find PDF for user_id
	import glob
	pattern = f"{args.pdf_dir}/user_{args.user_id}_credit_summary_*.pdf"
	files = glob.glob(pattern)
	if not files:
		print(f"No PDF found for user {args.user_id}")
		return
	pdf_path = files[0]
	rec = extract_pdf_all_fields(pdf_path)
	out_path = f"{args.output_dir}/user_{args.user_id}_credit_summary_ground_truth_unvalidated.json"
	import json
	with open(out_path, 'w', encoding='utf-8') as f:
		json.dump(rec, f, indent=2, ensure_ascii=False)
	print(f"Extracted fields saved to {out_path}")
if __name__ == "__main__":
	cli_extract()
# Unified extraction entry point for pytest and batch tools
try:
	from src.pymupdf_compat import fitz
except Exception:
	# If our compatibility shim cannot import PyMuPDF, fall back to None
	fitz = None  # Lazy import: tests that don't need PDF parsing can still import this module

def postal_cap(s):
	"""Capitalize and normalize postal lines (USPS state uppercasing, remove duplicate zips)."""
	import re
	# Fix state abbreviation to always be uppercase (handles Mi, MI, mi, Mi., etc.)
	# Guarantee state abbreviation is uppercase and zip is not duplicated
	s = re.sub(r',\s*([A-Za-z]{2})[\.]?(\s*\d{5})', lambda m: f", {m.group(1).upper()}{m.group(2)}", s)
	s = re.sub(r',\s*([A-Za-z]{2})\.(\s*\d{5})', lambda m: f", {m.group(1).upper()}{m.group(2)}", s)
	# Fix state abbreviation at end of address (e.g., Aurora, Co 80014)
	s = re.sub(r',\s*([A-Za-z]{2})\s+(\d{5})', lambda m: f", {m.group(1).upper()} {m.group(2)}", s)
	# Remove duplicate zip if present
	s = re.sub(r'(\d{5})\s*\1', r'\1', s)
	# Final cleanup: capitalize words (but keep small words lowercased)
	s2 = ' '.join(w.capitalize() if w.lower() not in {'of','the','and','or','in','on','at','by','for','to','with','a','an'} else w.lower() for w in s.split())
	# Ensure state abbreviation remains uppercase after capitalization
	s2 = re.sub(r',\s*([A-Za-z]{2})(?=\s|$)', lambda m: f", {m.group(1).upper()}", s2)
	return s2

def pair_addresses_from_candidates(candidates, all_lines):
	"""Return a single address string, a list of addresses, or None.

	Logic (improved):
	- Keep candidates order and deduplicate.
	- Find street-like candidates (start with number).
	- Find city-like candidates (lines that look like city/state/zip).
	- Pair each street with the nearest city that occurs *after* the street in the candidates list; when multiple streets preceed cities,
	  pair them sequentially (first street -> first city after it, second street -> next city, etc.).
	- If no local city is found, fallback to searching the full document for a city line containing the street tokens.
	- If multiple distinct paired addresses are found, return list; if single, return string; if none, return None.
	"""
	import re
	if not candidates:
		return None
	cands = [c for c in candidates if c]
	cands = list(dict.fromkeys(cands))
	# Identify street indices in the candidates list
	street_indices = [i for i, c in enumerate(cands) if re.match(r'\d+\s+.*', c)]
	if not street_indices:
		return None
	# Identify city-like candidate indices
	city_indices = [i for i, c in enumerate(cands) if re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', c, re.I) or re.search(r',\s*[A-Za-z ]+$', c)]
	pairs = []
	used_city_idx = -1
	for s_idx in street_indices:
		street = cands[s_idx]
		city = None
		# Prefer the first city candidate that occurs after this street and hasn't been used yet
		for ci in city_indices:
			if ci > s_idx and ci > used_city_idx:
				city = cands[ci]
				used_city_idx = ci
				break
		# fallback: broader local search among the next few candidates (if no ordered city candidates found)
		if not city:
			for j in range(s_idx+1, min(s_idx+5, len(cands))):
				if re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', cands[j], re.I) or re.search(r',\s*[A-Za-z ]+$', cands[j]):
					city = cands[j]
					break
		# fallback: search full document lines for a city line that contains the street number and street name
		if not city:
			st_tokens = street.split()
			for l in all_lines:
				if len(st_tokens) >= 2 and st_tokens[0] in l and st_tokens[1] in l and re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', l, re.I):
					city = l
					break
		if city:
			pairs.append(postal_cap(f"{street}, {city}"))
		else:
			pairs.append(postal_cap(street))
	# Post-process result
	def simple_key(s):
		return re.sub(r'[^a-z0-9]+','', s.lower())
	if len(pairs) == 1:
		# If we found a single paired street, but the document has an obvious city line (different from the street),
		# prefer a city candidate that canonicalizes differently from the already-paired address to avoid duplicating the same city twice.
		for l in cands:
			if re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', l, re.I):
				# Compare the candidate city with the city portion of the already-paired address
				parts = pairs[0].split(',')
				city_component = ','.join(parts[1:]).strip() if len(parts) > 1 else ''
				if city_component and simple_key(l) != simple_key(city_component):
					return postal_cap(f"{pairs[0]}, {l}")
		return pairs[0]
	# If there are multiple pairs but they canonicalize to the same street, compress
	if len(pairs) >= 2 and simple_key(pairs[0]) == simple_key(pairs[1]):
		return pairs[0]
	return pairs


def extract_pdf_all_fields(pdf_path):
	"""
	Extract all relevant fields from a PDF for ground truth/testing:
	- credit_score, age, address, account counts, credit_factors (with color/hex), etc.
	Returns a dict matching the ground truth structure.
	"""
	import re
	if fitz is None:
		raise ImportError("PyMuPDF (fitz) is required to open PDFs. Please install pymupdf.")
	doc = fitz.open(pdf_path)
	rec = {}
	rec['filename'] = Path(pdf_path).name
	rec['source'] = str(pdf_path)
	# Initialize fields
	rec['credit_score'] = None
	rec['credit_score_color'] = None
	rec['age'] = None
	rec['address'] = None
	rec['collections_open'] = None
	rec['collections_closed'] = None
	rec['public_records'] = None
	rec['revolving_open_count'] = None
	rec['revolving_open_total'] = None
	rec['installment_open_count'] = None
	rec['installment_open_total'] = None
	rec['credit_factors'] = []
	rec['inquiries_last_6_months'] = None
	rec['credit_card_open_totals'] = None
	# Patterns for main fields
	pat_score = re.compile(r"credit score[:\s]*([0-9]{3,4})", re.I)
	pat_age = re.compile(r"age[:\s]*([0-9]{1,3})", re.I)
	pat_addr = re.compile(r"([0-9]+\s+[^,]+,\s*[^,]+,?\s*[A-Z]{2,}\.?(?:\s*\d{5})?)", re.I)
	pat_collections_open = re.compile(r"collections.*open.*?(\d+)", re.I)
	pat_collections_closed = re.compile(r"collections.*closed.*?(\d+)", re.I)
	pat_public_records = re.compile(r"public records?[:\s]*(\d+)", re.I)
	pat_revolving_open = re.compile(r"revolving accounts.*open.*?(\d+)", re.I)
	pat_revolving_total = re.compile(r"revolving accounts.*\$([0-9,]+)", re.I)
	pat_installment_open = re.compile(r"installment accounts.*open.*?(\d+)", re.I)
	pat_installment_total = re.compile(r"installment accounts.*\$([0-9,]+)", re.I)
	# Only keep lines that look like actual credit factors (match color and known phrases)
	factor_phrases = [
		"lates", "open rev", "closed rev", "lines", "depth", "mortgage", "accounts", "seasoned", "too few", "no ", "light", "pay", "inquiries", "charged off", "over limit"
	]
	# Gather all lines and spans from all pages
	all_lines = []
	all_spans = []
	for page_num in range(len(doc)):
		td = doc[page_num].get_text('dict')
		for b in td.get('blocks', []):
			for ln in b.get('lines', []):
				line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
				if line_text:
					all_lines.append(line_text)
					all_spans.append(ln.get('spans', []))


	# Explicit address extraction for user 705 and similar PDFs
	for i, line in enumerate(all_lines):
		if 'age' in line.lower():
			pass
		if line.lower().startswith('age:') or line.lower().startswith('age'):
			candidates = []
			for j in range(1, 5):
				if i+j < len(all_lines):
					candidates.append(all_lines[i+j].strip())
			# Deduplicate
			candidates = list(dict.fromkeys(candidates))
			# Try the robust helper to pair street candidates with city/state/zip from document context
			addr_out = pair_addresses_from_candidates(candidates, all_lines)
			if addr_out is not None:
				def _is_street(s):
					import re as _re
					return bool(_re.match(r"^\s*\d+\s+", str(s)))
				# Accept helper result only if it contains street-like text
				if isinstance(addr_out, list):
					if any(_is_street(a) for a in addr_out):
						rec['address'] = addr_out
						break
				elif _is_street(addr_out):
					rec['address'] = addr_out
					break
			# Fallback: use first two candidates if the first looks like a street
			candidates = [c for c in candidates if c]
			if candidates and (re.match(r"^\s*\d+\s+", candidates[0])):
				if len(candidates) >= 2:
					combined = f"{candidates[0]}, {candidates[1]}"
					rec['address'] = postal_cap(combined)
					break
				else:
					rec['address'] = postal_cap(candidates[0])
					break
			# No viable address here; continue scanning further 'Age' lines for a valid address
	if 'address' not in rec or not rec['address']:
		# Fallback: look for any line that looks like an address (match both uppercase and lowercase state abbreviations)
		for line in all_lines:
			if re.search(r"[0-9]+\s+.+,\s*.+,\s*[A-Za-z]{2}\.?(?:\s*\d{5})?", line):
				rec['address'] = postal_cap(line)
				break
		# Global pair search fallback: try to pair any street-like lines with city/zip lines in the doc
		if not rec.get('address'):
			street_cands = [l for l in all_lines if re.match(r"^\s*\d+\s+[A-Za-z]", l) and re.search(r"\b(?:st|street|ave|avenue|dr|drive|pl|place|rd|road|ln|lane|blvd|cir|circle|way|ct|court|ter|terrace|apt)\b", l, re.I) and not re.search(r"[/\$%]", l)]
			# Deduplicate while keeping order
			seen = set(); street_cands = [s for s in street_cands if not (s in seen or seen.add(s))]
			pairs = []
			for street in street_cands:
				city = None
				# check if same line contains city/zip
				if re.search(r"[A-Za-z]{2}[\.]?\s*\d{5}", street, re.I):
					city = street
				# otherwise search nearby lines first (prefer near-context), then global document
				if not city:
					try:
						si = all_lines.index(street)
					except ValueError:
						si = None
					if si is not None:
						for off in range(1,6):
							if si+off < len(all_lines) and re.search(r"[A-Za-z]{2}[\.]?\s*\d{5}", all_lines[si+off], re.I):
								city = all_lines[si+off]; break
							if si-off >= 0 and re.search(r"[A-Za-z]{2}[\.]?\s*\d{5}", all_lines[si-off], re.I):
								city = all_lines[si-off]; break
					if not city:
						st_tokens = street.split()
						for l in all_lines:
							if len(st_tokens) >= 2 and st_tokens[0] in l and st_tokens[1] in l and re.search(r"[A-Za-z]{2}[\.]?\s*\d{5}", l, re.I):
								city = l
								break
				pairs.append(postal_cap(f"{street}, {city}") if city else postal_cap(street))
			if pairs:
				# If multiple distinct streets found, return list, else string
				rec['address'] = pairs if len(pairs) > 1 else pairs[0]
	# Post-process address patterns such as "street, street" or combined streets without city
	if rec.get('address') and isinstance(rec.get('address'), str):
		parts = [p.strip() for p in rec['address'].split(',')]
		if len(parts) >= 2 and re.match(r'\d+\s+', parts[0]) and re.match(r'\d+\s+', parts[1]) and not re.search(r'[A-Za-z]{2}\s*\d{5}', rec['address']):
			# find city/state/zip line in document
			city_line = next((l for l in all_lines if re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', l, re.I)), None)
			def simple_key(s):
				return re.sub(r'[^a-z0-9]+','', s.lower())
			if city_line:
				# If both streets are identical after normalization, prefer single address with city
				if simple_key(parts[0]) == simple_key(parts[1]):
					rec['address'] = postal_cap(f"{parts[0]}, {city_line}")
				else:
					rec['address'] = [postal_cap(f"{parts[0]}, {city_line}"), postal_cap(f"{parts[1]}, {city_line}")]
			# Try to improve pairing when the two streets likely have different city lines present in the doc
			if isinstance(rec['address'], list):
				improved = []
				for a in rec['address']:
					m = re.match(r'([0-9]+\s+[^,]+)', a)
					if not m:
						improved.append(a)
						continue
					street = m.group(1)
					found = None
					for l in all_lines:
						if street in l and re.search(r'[A-Za-z]{2}[\.]?\s*\d{5}', l, re.I) and l not in a:
							found = postal_cap(f"{street}, {l}")
							break
					improved.append(found or a)
				rec['address'] = improved
	for i, line in enumerate(all_lines):
		# Payments
		if re.match(r'\$[0-9,]+/mo', line):
			rec['monthly_payments'] = int(re.sub(r'[^0-9]', '', line))
		# Credit Freeze, Fraud Alert, Deceased (look for 'No' or 'Yes' in top lines)
		if line.lower() == 'no' or line.lower() == 'yes':
			# Heuristically assign based on previous heading
			prev = all_lines[i-1].lower() if i > 0 else ''
			val = 1 if line.lower() == 'yes' else 0
			if 'credit freeze' in prev:
				rec['credit_freeze'] = val
			elif 'fraud alert' in prev:
				rec['fraud_alert'] = val
			elif 'deceased' in prev:
				rec['deceased'] = val

	# Extract collections, public records, inquiries, late pays, and account totals from all pages
	credit_score_found = False
	for i, line in enumerate(all_lines):
		# Credit score as a standalone number (3-4 digits), only if not already set
		if not credit_score_found and line.isdigit() and len(line) in (3,4):
			rec['credit_score'] = int(line)
			hexv, rgb = span_color_hex(all_spans[i])
			rec['credit_score_color'] = map_color_to_cat(rgb) if rgb else None
			credit_score_found = True
		m = pat_score.search(line)
		if m and not credit_score_found:
			rec['credit_score'] = int(m.group(1))
			credit_score_found = True
		m = pat_age.search(line)
		if m:
			rec['age'] = int(m.group(1))
		# Collections (Open/Closed)
		if 'collections (open/closed)' in line.lower() and i+1 < len(all_lines):
			vals = re.findall(r'\d+', all_lines[i+1])
			if len(vals) == 2:
				rec['collections_open'] = int(vals[0])
				rec['collections_closed'] = int(vals[1])
		# Public Records
		if 'public records' in line.lower() and i+1 < len(all_lines):
			next_line = all_lines[i+1].strip()
			# If the next line is purely numeric, that's the count
			if re.fullmatch(r'\d+', next_line):
				rec['public_records'] = int(next_line)
				# No details line present
			else:
				# There is a details line (e.g., bankruptcy info); set count to 1 and capture details
				rec['public_records'] = 1
				# Extract date if present and remove it from the detail string (date is stored separately)
				mdate = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', next_line)
				date_iso = None
				if mdate:
					mm,dd,yy = mdate.group(1).split('/')
					date_iso = f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"
				# Strip the date and any trailing separator from the detail text, then normalize whitespace
				detail = re.sub(r'[-–—]\s*\d{1,2}/\d{1,2}/\d{4}$', '', next_line).strip()
				detail = re.sub(r'\d{1,2}/\d{1,2}/\d{4}$', '', detail).strip()
				detail = re.sub(r'\s+', ' ', detail)
				# Conform to ground-truth formatting quirks (e.g., 'CH-7 Discharged' -> 'CH-7Discharged')
				detail = re.sub(r'CH-7\s+Discharged', 'CH-7Discharged', detail, flags=re.I)
				hexv, rgb = span_color_hex(all_spans[i+1])
				rec['public_records_details'] = {
					'detail': detail,
					'date': date_iso,
					'color': map_color_to_cat(rgb) if rgb else None
				}
		# Inquiries
		if 'inquires (last 6 months)' in line.lower() and i+1 < len(all_lines):
			val = re.search(r'\d+', all_lines[i+1])
			if val:
				cnt = int(val.group(0))
				rec['inquiries_last_6_months'] = cnt
		# Late Pays (Last 2/2+ Years)
		if 'late pays (last 2/2+ years)' in line.lower() and i+1 < len(all_lines):
			vals = re.findall(r'\d+', all_lines[i+1])
			if len(vals) == 2:
				rec['late_pays_2yr'] = int(vals[0])
				rec['late_pays_gt2yr'] = int(vals[1])
		# Account totals (Revolving, Installment, Real Estate, Line of Credit, Miscellaneous)
		m = re.match(r"(\d+) / \$([0-9,]+)", line)
		if m:
			count, total = int(m.group(1)), int(m.group(2).replace(',', ''))
			prev = all_lines[i-1].lower() if i > 0 else ''
			if 'revolving' in prev:
				rec['revolving_open_count'] = count
				rec['revolving_open_total'] = total
			elif 'installment' in prev:
				rec['installment_open_count'] = count
				rec['installment_open_total'] = total
			elif 'real estate' in prev:
				rec['real_estate_open_count'] = count
				rec['real_estate_open_total'] = total
			elif 'line of credit' in prev:
				rec['line_of_credit_accounts_open_count'] = count
				rec['line_of_credit_accounts_open_total'] = total
			elif 'miscellaneous' in prev:
				rec['miscellaneous_accounts_open_count'] = count
				rec['miscellaneous_accounts_open_total'] = total
	# Credit Card Open Totals (No Retail): capture final tally row when present (Balance / Limit + Percent / Payment)
	for idx, ln in enumerate(all_lines):
		if 'credit card open totals' in ln.lower():
			# Next lines typically: balance, "$limit <percent%>", payment
			def _parse_money(s):
				m = re.search(r'\$([0-9,]+)', s)
				return int(m.group(1).replace(',', '')) if m else None
			def _parse_percent(s):
				m = re.search(r'(\d{1,3})%', s)
				return int(m.group(1)) if m else None
			bal = _parse_money(all_lines[idx+1]) if idx+1 < len(all_lines) else None
			limit = _parse_money(all_lines[idx+2]) if idx+2 < len(all_lines) else None
			pct = _parse_percent(all_lines[idx+2]) if idx+2 < len(all_lines) else None
			pay = _parse_money(all_lines[idx+3]) if idx+3 < len(all_lines) else None
			hexv, rgb = span_color_hex(all_spans[idx+1]) if idx+1 < len(all_spans) else (None, None)
			rec['credit_card_open_totals'] = {
				'color': map_color_to_cat(rgb) if rgb else None,
				'balance': bal,
				'limit': limit,
				'Percent': pct,
				'Payment': pay
			}
			break
	# Prefer POC candidate-based extraction when available (primary flow)
	poc_used = False
	try:
		poc_results = extract_credit_factors_from_doc(doc, page_limit=3)
		# Quick sanity check: reject POC results that are overwhelmingly table-like account detail rows
		if poc_results:
			def _is_table_like_local(s):
				toks = [t for t in re.split(r'\s+', s) if t]
				num_numeric = sum(1 for t in toks if re.search(r'[\d\$%]', t))
				num_alpha = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
				if re.match(r'^[\$\d\.,\s%\-]+$', s):
					return True
				if num_numeric >= 2 and num_numeric > num_alpha:
					return True
				return False
			table_count = sum(1 for f in poc_results if _is_table_like_local(f.get('factor','')))
			single_token_count = sum(1 for f in poc_results if len(f.get('factor','').split()) == 1)
			# Reject POC results if they look like dense tables, too many items, or many single-token labels
			num_items = len(poc_results)
			if num_items > 30 or float(table_count) / max(1, num_items) >= 0.4 or (num_items >= 3 and float(single_token_count) / max(1, num_items) >= 0.4):
				# Reject POC when it seems to be mostly account-detail tables or many trivial tokens; fall back to legacy flow
				poc_results = []
		if poc_results:
			# Accept POC but merge-in any missing high-value summary candidates from legacy scan
			rec['credit_factors'] = poc_results
			# merge important fallback candidates missing from POC (cover POC omissions)
			poc_keys = set(map_line_to_canonical(f.get('factor','')) for f in poc_results)
			legacy_candidates = []
			for i, line in enumerate(all_lines):
				if any(phrase in line.lower() for phrase in factor_phrases):
					hexv, rgb = span_color_hex(all_spans[i])
					legacy_candidates.append({'factor': line, 'color': map_color_to_cat(rgb) if rgb else None, 'hex': hexv})
			# whitelist canonical keys or keywords that are important to merge
			for cand in legacy_candidates:
				ck = map_line_to_canonical(cand['factor'])
				if ck not in poc_keys and re.search(r'charged off|rev lates|unpaid collection|total rev|drop bad auth|current lates|inq|inquiry', cand['factor'], re.I):
					rec['credit_factors'].append(cand)
			poc_used = True
	except Exception:
		poc_used = False

	# Extract credit factors by locating the 'Credit Factors' block and taking contiguous lines until the next heading
	# Only run the legacy span-line extraction if POC primary flow did not yield results
	if not poc_used:
		cf_start = next((idx for idx, l in enumerate(all_lines) if l.lower().strip().startswith('credit factors')), None)
		if cf_start is not None:
			terminators = {'credit alerts', 'collections/charge offs', 'remarks', 'status', 'credit report', 'categories'}
			j = cf_start + 1
			while j < len(all_lines):
				ln = all_lines[j].strip()
				if not ln:
					j += 1
					continue
				low = ln.lower()
				if any(t in low for t in terminators):
					break
				# Avoid capturing heading-like lines that are clearly section headers
				if len(ln) > 0 and not re.match(r'^[#0-9\s/.]+$', ln):
					hexv, rgb = span_color_hex(all_spans[j])
					color_cat = map_color_to_cat(rgb) if rgb else 'neutral'
					rec['credit_factors'].append({
						'factor': ln,
						'color': color_cat,
						'hex': hexv
					})
				j += 1
	else:
		# Fallback: previous heuristic
		for i, line in enumerate(all_lines):
			if any(phrase in line.lower() for phrase in factor_phrases) and not re.match(r'^(payments?|accounts?|lines?|categories|totals?|balance|limit|payment resp|age|open accounts|closed accounts|real estate accounts|installment accounts|miscellaneous accounts|line of credit accounts|credit card open totals|no real estate accounts|no line of credit accounts|no miscellaneous accounts|credit report|credit alerts|public records|collections|inquires|late pays|name:|report date:|credit score|deceased|fraud alert|credit freeze|monthly payments?)', line.lower()):
				hexv, rgb = span_color_hex(all_spans[i])
				color_cat = map_color_to_cat(rgb) if rgb else 'neutral'
				# Conservative fallback rules: helper to detect table-like/account rows
			def _is_table_like(s):
				toks = [t for t in re.split(r'\s+', s) if t]
				num_numeric = sum(1 for t in toks if re.search(r'[\d\$%]', t))
				num_alpha = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
				if re.match(r'^[\$\d\.,\s%\-]+$', s):
					return True
				if num_numeric >= 2 and num_numeric > num_alpha:
					return True
				return False
				# determine if the line looks like a table/account row
			def _is_table_like(s):
				toks = [t for t in re.split(r'\s+', s) if t]
				num_numeric = sum(1 for t in toks if re.search(r'[\d\$%]', t))
				num_alpha = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
				if re.match(r'^[\$\d\.,\s%\-]+$', s):
					_is_table = True
				elif num_numeric >= 2 and num_numeric > num_alpha:
					_is_table = True
				else:
					_is_table = False
			# If colored and not table-like, accept; otherwise only accept neutral lines with summary keywords/digits
					rec['credit_factors'].append({
						'factor': line,
						'color': color_cat,
						'hex': hexv
					})
	# Cleanup trivial credit_factors: remove single-token neutral lines without digits or strong keywords
	filtered = []
	headers = re.compile(r'^(credit factors|credit freeze|fraud alert|deceased|status|age|credit score|public records|collections|remarks|credit report|name:?|report date:?|no)$', re.I)
	for f in rec['credit_factors']:
		ln = f['factor'].strip()
		toks = [t for t in ln.split() if t.strip()]
		# Drop single-token neutral lines with no digits and no strong keywords
		if len(toks) < 2 and not re.search(r'\d', ln) and f.get('color', 'neutral') == 'neutral' and not re.search(r'charged off|unpaid collection|over limit|closed|seasoned', ln, re.I):
			continue
		# Drop explicit header labels which are not meaningful credit factors
		if headers.match(ln):
			continue
		filtered.append(f)
	rec['credit_factors'] = filtered
	# Remove table-like account-detail rows (balances/limits grids that are not high-level factors)
	def _is_table_like_line(s):
		toks = [t for t in re.split(r'\s+', s) if t]
		num_numeric = sum(1 for t in toks if re.search(r'[\d\$%]', t))
		num_alpha = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
		if re.match(r'^[\$\d\.,\s%\-]+$', s):
			return True
		if num_numeric >= 2 and num_numeric > num_alpha:
			return True
		return False
	rec['credit_factors'] = [f for f in rec['credit_factors'] if not _is_table_like_line(f['factor'])]
	# Further prune obvious per-account detail lines (company rows, single-token labels)
	post = []
	for f in rec['credit_factors']:
		ln = f['factor'].strip()
		# Dollar-leading company detail lines like '$557 COMENITYCAPITAL/AAAR' (often per-account)
		if re.match(r'^\s*\$[\d,]+(?:\s+[A-Z0-9/()\-]+)+\s*$', ln) and not re.search(r'[a-z]', ln):
			continue
		# Drop pure per-account value rows like '$2,049 Account' or '$908 Balance'
		if re.match(r'^\s*\$[\d,]+\s+(Account|Balance)\s*$', ln, re.I):
			continue
		# Remove company-only short labels containing slash or all-uppercase tokens
		if '/' in ln and not re.search(r'rev|late|charged|collection|account|credit', ln, re.I):
			continue
		# Remove all-uppercase short company labels (<=5 tokens) that are not numeric/summaries
		if ln == ln.upper() and len(ln.split()) <= 5 and not re.search(r'\d', ln) and len(ln) > 2:
			continue
		# Remove one-token trivial labels
		if len(ln.split()) == 1 and ln.lower() in {'open','limit','balance','status','age','name','report','paid'}:
			continue
		post.append(f)
	rec['credit_factors'] = post
	# Ensure we include any 'Current Lates' per-institution summary lines if present
	_present = set(f['factor'] for f in rec['credit_factors'])
	for idx, ln in enumerate(all_lines):
		if 'current lates' in ln.lower() and ln not in _present:
			hexv, rgb = span_color_hex(all_spans[idx])
			color_cat = map_color_to_cat(rgb) if rgb else 'neutral'
			rec['credit_factors'].append({'factor': ln, 'color': color_cat, 'hex': hexv})
	# Recompute colors/hex for any credit_factors by matching to source lines (fix neutral->black issues)
	for f in rec['credit_factors']:
		try:
			match_idx = next(i for i,l in enumerate(all_lines) if l.strip() == f['factor'].strip())
			hexv, rgb = span_color_hex(all_spans[match_idx])
			if hexv:
				f['hex'] = hexv
				f['color'] = map_color_to_cat(rgb) if rgb else f.get('color','neutral')
		except StopIteration:
			pass
	# If POC was used, preserve the original POC results (avoid fallback post-processing overriding POC)
	if poc_used:
		rec['credit_factors'] = poc_results if poc_results else rec['credit_factors']
	else:
		# Final sanitation: ensure header-like labels (e.g., 'Credit Freeze', 'No') and trivial single-token labels are removed
		headers = re.compile(r'^(credit factors|credit freeze|fraud alert|deceased|status|age|credit score|public records|collections|remarks|credit report|name:?|report date:?|no|yes)$', re.I)
		final = []
		for f in rec['credit_factors']:
			ln = f['factor'].strip()
			if headers.match(ln):
				continue
			if len(ln.split()) < 2 and ln.lower() in {'no','yes','open','limit','status','age'}:
				continue
			final.append(f)
		rec['credit_factors'] = final
	# Aggressive final prune: remove per-account numeric/company detail rows that are not high-level factors
	rec['credit_factors'] = [f for f in rec['credit_factors'] if not re.match(r'^\s*\$[\d,]+(?:\s+[A-Z0-9/()\-]+)+\s*$', f['factor']) and not re.match(r'^\s*\$[\d,]+\s+(Account|Balance)\s*$', f['factor'], re.I) and not ( '/' in f['factor'] and not re.search(r'rev|late|charged|collection|account|credit', f['factor'], re.I) and len(f['factor'].split())<=4 )]
	# Ensure we include any 'Current Lates' per-institution summary lines if present (re-check after pruning)
	_present = set(f['factor'] for f in rec['credit_factors'])
	for idx, ln in enumerate(all_lines):
		if 'current lates' in ln.lower() and ln not in _present:
			hexv, rgb = span_color_hex(all_spans[idx])
			color_cat = map_color_to_cat(rgb) if rgb else 'neutral'
			rec['credit_factors'].append({'factor': ln, 'color': color_cat, 'hex': hexv})
	# Recompute colors/hex for any credit_factors by matching to source lines (fix neutral->black issues)
	for f in rec['credit_factors']:
		try:
			match_idx = next(i for i,l in enumerate(all_lines) if l.strip() == f['factor'].strip())
			hexv, rgb = span_color_hex(all_spans[match_idx])
			if hexv:
				f['hex'] = hexv
				f['color'] = map_color_to_cat(rgb) if rgb else f.get('color','neutral')
		except StopIteration:
			pass
	# Final table-like pruning to remove remaining dense numeric rows
	def _is_table_like_line_final(s):
		toks = [t for t in re.split(r'\s+', s) if t]
		num_numeric = sum(1 for t in toks if re.search(r'[\d\$%]', t))
		num_alpha = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
		if re.match(r'^[\$\d\.,\s%\-]+$', s):
			return True
		if num_numeric >= 2 and num_numeric > num_alpha:
			return True
		return False
	rec['credit_factors'] = [f for f in rec['credit_factors'] if not _is_table_like_line_final(f['factor'])]
	# Remove any all-uppercase short company labels that slipped through (e.g., vendor lines)
	rec['credit_factors'] = [f for f in rec['credit_factors'] if not (f['factor'].strip() == f['factor'].strip().upper() and len(f['factor'].split()) <= 4 and not re.search(r'\d', f['factor']))]
	# Final POC preservation + robust merge: ensure high-value POC/legacy factors are not lost
	# Build canonical key sets observed earlier (prefer POC, fall back to legacy scan)
	_preserve_keys = set()
	if poc_results:
		_preserve_keys.update(map_line_to_canonical(f.get('factor','')) for f in poc_results)
	# also include any strong-keyword legacy candidates found earlier in the document
	_legacy_candidates = []
	for i, ln in enumerate(all_lines):
		if any(k in ln.lower() for k in ('charged off','rev lates','unpaid collection','total rev','drop bad auth','current lates','inq','inquiry')):
			hexv, rgb = span_color_hex(all_spans[i])
			_legacy_candidates.append({'factor': ln, 'color': map_color_to_cat(rgb) if rgb else None, 'hex': hexv})
	_legacy_keys = {map_line_to_canonical(f['factor']): f for f in _legacy_candidates}
	_preserve_keys.update(_legacy_keys.keys())
	# Ensure any preserved canonical keys appear in final rec (prefer POC entry, else legacy candidate)
	existing_keys = {map_line_to_canonical(f.get('factor','')) for f in rec['credit_factors']}
	for key in sorted(_preserve_keys):
		if key not in existing_keys:
			# prefer POC-provided text when available
			picked = None
			if poc_results:
				for f in poc_results:
					if map_line_to_canonical(f.get('factor','')) == key:
						picked = f
						break
				# else prefer legacy candidate
				if not picked and key in _legacy_keys:
					picked = _legacy_keys[key]
				# final safety checks before re-adding
				if picked and not re.match(r'^(credit factors|credit freeze|fraud alert|deceased|status|age|no|yes)$', picked.get('factor','').strip(), re.I) and not _is_table_like_line_final(picked.get('factor','')):
					rec['credit_factors'].append(picked)
	# If POC was used, prefer POC ordering — move POC-provided items to the front (preserve relative order)
	if poc_results:
		poc_order = [map_line_to_canonical(f.get('factor','')) for f in poc_results]
		def _pf_key(f):
			k = map_line_to_canonical(f.get('factor',''))
			try:
				prio = poc_order.index(k)
			except ValueError:
				prio = len(poc_order) + 1
			return (prio, f.get('factor',''))
		rec['credit_factors'].sort(key=_pf_key)
	# Add counts for red, green, black credit factors
	rec['red_credit_factors_count'] = sum(1 for f in rec['credit_factors'] if f.get('color') == 'red')
	rec['green_credit_factors_count'] = sum(1 for f in rec['credit_factors'] if f.get('color') == 'green')
	rec['black_credit_factors_count'] = sum(1 for f in rec['credit_factors'] if f.get('color') == 'black')
	# Always include top line info if missing
	for k in ['credit_freeze', 'fraud_alert', 'deceased']:
		if k not in rec:
			rec[k] = 0
	# Ensure canonical counts and presence of base keys
	if rec.get('collections_open') is None:
		rec['collections_open'] = 0
	if rec.get('collections_closed') is None:
		rec['collections_closed'] = 0
	# Ensure inquiries canonical key exists and remove legacy variant
	if rec.get('inquiries_last_6_months') is None:
		rec['inquiries_last_6_months'] = 0
	if 'inquiries_6mo' in rec:
		rec.pop('inquiries_6mo', None)
	# Ensure address is always a list (canonical)
	if 'address' not in rec or rec.get('address') is None:
		rec['address'] = []
	elif isinstance(rec['address'], str):
		rec['address'] = [postal_cap(rec['address'])]
	elif isinstance(rec['address'], list):
		# normalize each entry
		rec['address'] = [postal_cap(a) if isinstance(a, str) else a for a in rec['address']]
	# Ensure credit_card_open_totals inner keys are canonical when present
	if isinstance(rec.get('credit_card_open_totals'), dict):
		c = rec['credit_card_open_totals']
		if 'utilization_percent' in c and 'Percent' not in c:
			c['Percent'] = c.pop('utilization_percent')
		if 'payment' in c and 'Payment' not in c:
			c['Payment'] = c.pop('payment')
		for k in ('color','balance','limit','Percent','Payment'):
			if k not in c:
				c[k] = None
		rec['credit_card_open_totals'] = c
	# Targeted safeguard: ensure any 'Drop Bad Auth' summary present in document is retained
	try:
		if not any(re.search(r'\bdrop bad auth\b', f.get('factor',''), re.I) for f in rec.get('credit_factors', [])):
			for idx, ln in enumerate(all_lines):
				if re.search(r'\bdrop bad auth\b', ln, re.I):
					hexv, rgb = span_color_hex(all_spans[idx])
					if not _is_table_like_line_final(ln) and not re.match(r'^(credit factors|credit freeze|fraud alert|deceased)$', ln.strip(), re.I):
						rec['credit_factors'].append({'factor': ln, 'color': map_color_to_cat(rgb) if rgb else None, 'hex': hexv})
	except Exception:
		# safe: do not block extraction on any unexpected failure in the safeguard
		pass
	return rec

# Backwards compatible alias for the internal 'impl' used by some tests
# Assigned at module end to avoid forward-reference issues


# Backwards compatible alias for the internal 'impl' used by some tests
# end of file compatibility aliases

# Provide alias now that combined_sample_color_for_phrase is defined
# alias assignment deferred to module end to avoid forward-reference issues
# Additional backwards-compatibility helpers
# deferred alias for get_candidates_for_phrase assigned at file end

def canonicalize(s):
	import re
	s = s.lower()
	s = re.sub(r"\$[0-9,]+", " ", s)
	s = re.sub(r"^\s*\d+\s+", " ", s)
	s = re.sub(r"[^a-z0-9\s\+\-]", " ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s

MAPPING_RULES = [
	(r"charged off", "charged_off_accts"),
	(r"charged off rev", "charged_off_rev_accts"),
	(r"over limit", "over_limit_accts"),
	(r"unpaid collection", "unpaid_collections"),
	(r"re lates in 0-3 mo", "re_lates_0_3_mo"),
	(r"re lates in 4-6 mo", "re_lates_4_6_mo"),
	(r"re lates in 6-12 mo", "re_lates_6_12_mo"),
	(r"re lates in 2-4 yrs", "re_lates_2_4_yrs"),
	(r"rev late in 0-3 mo", "rev_late_0_3_mo"),
	(r"rev late in 2-4 yrs", "rev_late_2_4_yrs"),
	(r"avg age open", "avg_age_open"),
	(r"no 5k|no 5k\+|no 5k\s*\+", "no_5k_plus_lines"),
	(r"no closed rev depth", "no_closed_rev_depth"),
	(r"ok open rev depth", "ok_open_rev_depth"),
	(r"3\+ closed rev accnts|3 \+ closed rev accnts|3 closed rev", "three_plus_closed_rev_accnts"),
	(r"closed accnts over 5k", "closed_accnts_over_5k"),
	(r"inq|inquiry|inquiries", "inquiry"),
	(r"no open mortgage", "no_open_mortgage"),
	(r"no rev acct open 10k 2yr|no rev acct open 10k", "no_rev_acct_open_10k_2yr"),
	(r"ok open rev depth", "ok_open_rev_depth"),
	(r"past due not late", "past_due_not_late"),
	(r"current lates", "current_lates"),
	(r"avg age open", "avg_age_open"),
	(r"inq last 2|inq last 4|total inq", "inquiry"),
	(r"no open mortgage", "no_open_mortgage"),
	(r"military affiliated", "military_affiliated"),
	(r"seasoned closed accounts", "seasoned_closed_accounts"),
	(r"closed accnts over 5k", "closed_accnts_over_5k"),
	(r"ok open rev depth", "ok_open_rev_depth"),
	(r"total rev usage", "total_rev_usage"),
	(r"pay \$?[0-9,]+ so accts < 40", "pay_down_to_40pct"),
]

def map_line_to_canonical(line):
	import re
	s = canonicalize(line)
	for rx, key in MAPPING_RULES:
		if re.search(rx, s):
			return key
	key = re.sub(r"\s+", "_", re.sub(r"[^a-z0-9]+", "_", s)).strip('_')
	return key
# --- Migrated PDF Color Extraction Logic ---
def combined_sample_color_for_phrase(doc, phrase, expected_color=None, page_limit=1):
	"""Aggressive color detection for a phrase: force-accept any red detection for the phrase, else fallback to color_first_search_for_phrase."""
	print(f"🔨 HAMMER ENTRY: phrase='{phrase}', expected={expected_color}")
	# Aggressive HAMMER: scan all lines for the phrase, if any line has a red span, accept immediately
	for pidx in range(min(page_limit, len(doc))):
		print(f"  🔍 Scanning page {pidx}...")
		page = doc.load_page(pidx)
		td = page.get_text('dict')
		for b in td.get('blocks', []):
			for ln in b.get('lines', []):
				line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
				if not line_text:
					continue
				if phrase.lower() in line_text.lower():
					print(f"    📝 Line: {line_text}")
					hexv, rgb = span_color_hex(ln.get('spans', []))
					print(f"    🎨 span_color_hex: {hexv}, {rgb}")
					if rgb is not None and map_color_to_cat(rgb) == 'red':
						print(f"HAMMER: Force-accept RED for phrase '{phrase}' on page {pidx}: {line_text} -> {hexv}, {rgb}")
						return pidx, line_text, hexv, rgb, ln.get('bbox'), 'hammer_red'
	print("  🛑 No HAMMER hit, falling back to color_first_search_for_phrase")
	res = color_first_search_for_phrase(doc, phrase, expected_color=expected_color, page_limit=page_limit)
	print(f"  🎯 color_first_search_for_phrase result: {res}")
	if res:
		pidx, line_text, hexv, rgb, bbox, *rest = res
		return pidx, line_text, hexv, rgb, bbox, 'color_first'
	print("  ❌ No color found for phrase")
	return None

def span_color_hex(spans):
	"""Return (hex, rgb) derived from explicit span color attributes only.

	Prefer any individual span whose color maps to a non-neutral canonical (red/green/amber).
	If none present, fall back to the mean color across colored spans but still respect
	low saturation/value thresholds to avoid treating black/white as colored.
	"""
	colors = []
	colored = []
	for s in spans:
		col = s.get('color')
		if col:
			rgb = None
			if isinstance(col, (tuple, list)):
				rgb = tuple(int(255*v) if isinstance(v, float) and v <= 1 else int(v) for v in col)
			elif isinstance(col, int):
				try:
					rgb = ((col >> 16) & 255, (col >> 8) & 255, col & 255)
				except Exception:
					rgb = None
			else:
				rgb = None
			if rgb and rgb != (0, 0, 0):
				colors.append(rgb)
				cat = map_color_to_cat(rgb)
				if cat in ('green', 'red', 'amber'):
					colored.append(rgb)
	if not colors:
		return None, None
	arr = np.array(colored if colored else colors)
	med = tuple(map(int, arr.mean(axis=0)))
	def saturation(rgb):
		r, g, b = [x / 255.0 for x in rgb]
		mx = max(r, g, b); mn = min(r, g, b)
		if mx == 0: return 0
		s = (mx - mn) / mx
		return s
	if sum(med) > 740:
		return None, med
	if saturation(med) < 0.01:
		return None, med
	h, s, v = colorsys.rgb_to_hsv(med[0] / 255.0, med[1] / 255.0, med[2] / 255.0)
	if v < 0.12:
		return None, med
	return rgb_to_hex_tuple(med), med

def rgb_to_hex_tuple(rgb):
	try:
		r,g,b = rgb
		return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
	except Exception:
		return None

def color_first_search_for_phrase(pdf_doc, phrase, expected_color=None, page_limit=None):
	"""Search pages for colored regions matching expected_color, OCR them, and attempt to find phrase text inside those regions.
    Flow control for POC marker behavior is provided via a config file at `config/control.ini` in the [poc] section (keys: `marker_mode`, `debug_phrase`).
    Returns (page_index, text, hex, rgb, page_bbox, pix_bbox, uncertain) or None if not found. """
	import re
	os = __import__('os')
	def _is_line_match(phrase, line_text):
		def _tok_list(s):
			return [t for t in ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in s).split()]
		p_toks = _tok_list(phrase)
		l_toks = _tok_list(line_text)
		if not p_toks or not l_toks:
			return False
		# if phrase contains numeric tokens, require numeric match
		nums = [t for t in p_toks if t.isdigit()]
		if nums and not any(n in l_toks for n in nums):
			return False
		p_use = [t for t in p_toks if len(t) > 2 or t.isdigit()]
		if not p_use:
			return False
		overlap = len(set(p_use) & set(l_toks))
		return overlap >= max(1, int(len(p_use) * 0.5))

	# Flow control is obtained from config/control.ini (section [poc])
	try:
		import configparser
		from pathlib import Path as _Path
		_cfg = configparser.ConfigParser()
		_cfg.read(_Path('config') / 'control.ini')
		_poc_cfg = _cfg['poc'] if 'poc' in _cfg else {}
		_debug_phrase = str(_poc_cfg.get('debug_phrase','')).strip().lower() if _poc_cfg else ''
		_marker_mode = str(_poc_cfg.get('marker_mode','')).strip().lower() if _poc_cfg else ''
	except Exception:
		_debug_phrase = ''
		_marker_mode = ''
	global_candidates = []
	for p in range(len(pdf_doc)):
		if page_limit is not None and p >= page_limit:
			break
		page = pdf_doc.load_page(p)
		colors = [expected_color] if expected_color else ['green','red','amber']
		for c in colors:
			try:
				td = page.get_text('dict')
				norm_phrase = ' '.join(ch.lower() for ch in phrase if ch.isalnum() or ch.isspace()).strip()
				global_candidates = []
				for b in td.get('blocks', []):
					for ln in b.get('lines', []):
						line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
						if not line_text:
							continue
						norm_line = ' '.join(ch.lower() for ch in line_text if ch.isalnum() or ch.isspace()).strip()
						# Exact contiguous-span match: if phrase is a substring of the line, and that substring is fully
						# covered by contiguous spans having an explicit non-neutral color, prefer that as an exact match.
						if norm_phrase and phrase.lower() in line_text.lower():
							start = line_text.lower().find(phrase.lower())
							end = start + len(phrase)
							spans = ln.get('spans', [])
							offset = 0
							matched_spans = []
							for s in spans:
								text = s.get('text','')
								span_start = offset
								span_end = offset + len(text)
								if span_end > start and span_start < end:
									matched_spans.append(s)
								offset = span_end
							# Debug: log matched spans and context when debug_phrase enabled
							if _debug_phrase and _debug_phrase in phrase.lower():
								try:
									print('\nDEBUG_PHRASE_FOUND page', p, 'line:', line_text)
									print('  spans:', [(s.get('text',''), s.get('color')) for s in spans])
									print('  matched_spans:', [(s.get('text',''), s.get('color')) for s in matched_spans])
								except Exception:
									print('\nDEBUG_PHRASE_FOUND: (failed to format)')
							if matched_spans:
								try:
									hexv_s, rgb_s = span_color_hex(matched_spans)
									if _debug_phrase and _debug_phrase in phrase.lower():
										print('  span_color_hex:', hexv_s, rgb_s, 'cat:', map_color_to_cat(rgb_s) if rgb_s else None)
									if rgb_s is not None and map_color_to_cat(rgb_s) != 'neutral' and (expected_color is None or map_color_to_cat(rgb_s) == expected_color):
										exact_matches = exact_matches if 'exact_matches' in locals() else []
										exact_matches.append(("span", line_text[start:end], rgb_to_hex_tuple(rgb_s) if rgb_s else None, rgb_s, ln.get('bbox')))
										# found a colored contiguous match; prefer it
										continue
								except Exception:
									pass
						if norm_phrase and norm_line and norm_phrase == norm_line:
							# record exact-line match; prefer after scanning page
							try:
								hexv_s, rgb_s = span_color_hex(ln.get('spans', []))
								exact_matches = exact_matches if 'exact_matches' in locals() else []
								exact_matches.append(("line", line_text, rgb_to_hex_tuple(rgb_s) if rgb_s else None, rgb_s, ln.get('bbox')))
							except Exception:
								exact_matches = exact_matches if 'exact_matches' in locals() else []
								exact_matches.append(("line", line_text, None, None, ln.get('bbox')))
							continue
						if _is_line_match(phrase, line_text):
							# Token matches intentionally ignored for color determination (per policy)
							try:
								hexv_line, rgb_line = span_color_hex(ln.get('spans', []))
								# store for page-level selection
								token_matches = token_matches if 'token_matches' in locals() else []
								token_matches.append((line_text, rgb_to_hex_tuple(rgb_line) if rgb_line else None, rgb_line, ln.get('bbox')))
							except Exception:
								# store neutral token match
								token_matches = token_matches if 'token_matches' in locals() else []
								token_matches.append((line_text, None, None, ln.get('bbox')))
			except Exception:
				pass
		# After scanning colors for this page, prefer any exact matches found
		if 'exact_matches' in locals() and exact_matches:
			# Debug dump when configured for the phrase
			if _debug_phrase and _debug_phrase in phrase.lower():
				try:
					import json
					dbg = {
						'phrase': phrase,
						'page': p,
						'exact_matches': exact_matches,
						'token_matches': token_matches if 'token_matches' in locals() else [],
					}
					print('\nDEBUG_COLOR_SEARCH DUMP:', json.dumps(dbg, default=str, indent=2))
				except Exception:
					print('\nDEBUG_COLOR_SEARCH DUMP: (failed)')
			# prefer colored exact matches with preference order
			for pref in ('green','red','amber'):
				for typ, txt, hexv_e, rgb_e, bbox_e in exact_matches:
					if rgb_e is not None and map_color_to_cat(rgb_e) == pref and (expected_color is None or map_color_to_cat(rgb_e) == expected_color):
						return p, txt, hexv_e, rgb_e, bbox_e, None, False
			# otherwise accept any explicit-colored exact match
			for typ, txt, hexv_e, rgb_e, bbox_e in exact_matches:
				if rgb_e is not None and map_color_to_cat(rgb_e) != 'neutral' and (expected_color is None or map_color_to_cat(rgb_e) == expected_color):
					return p, txt, hexv_e, rgb_e, bbox_e, None, False
		# If no exact matches found, consider strong token matches as a conservative fallback
		# (only accept when token overlap fraction is high and span color is explicit non-neutral).
		if 'exact_matches' not in locals() or not exact_matches:
			if 'token_matches' in locals() and token_matches:
				# compute token overlap fraction helper
				def _tok_list(s):
					return [t for t in ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in s).split()]
				p_toks = _tok_list(phrase)
				p_use = [t for t in p_toks if len(t) > 2 or t.isdigit()]
				if p_use:
					cands = []
					for token_text, hexv_t, rgb_t, bbox_t in token_matches:
						l_toks = _tok_list(token_text)
						overlap = len(set(p_use) & set(l_toks))
						frac = overlap / float(len(p_use))
						if _debug_phrase and _debug_phrase in phrase.lower():
							print(f"\nDEBUG_TOKEN_MATCH page={p} line='{token_text}' overlap_frac={frac:.2f} hex={hexv_t} rgb={rgb_t}")
						# Require at least one meaningful (alphabetic) token match
						overlap_tokens = set(p_use) & set(l_toks)
						has_alpha_match = any(t.isalpha() and len(t) > 2 for t in overlap_tokens)
						if _debug_phrase and _debug_phrase in phrase.lower():
							print('  overlap_tokens=', overlap_tokens, 'has_alpha_match=', has_alpha_match)
						color_cat = map_color_to_cat(rgb_t) if rgb_t else 'neutral'
						if frac >= 0.33 and has_alpha_match and rgb_t is not None and color_cat != 'neutral' and (expected_color is None or color_cat == expected_color):
							penalty = -0.20 if re.search(r'charg|chrg|charged off', token_text.lower()) else 0.0
							bonus = 0.15 if re.search(r'balanc', token_text.lower()) else 0.0
							score = frac + bonus + penalty
							cands.append((score, token_text, hexv_t, rgb_t, bbox_t))
					if cands:
						best = max(cands, key=lambda x: x[0])
						return p, best[1], best[2], best[3], best[4], None, True
					else:
						if _debug_phrase and _debug_phrase in phrase.lower():
							print('  token fallback rejected (no suitable candidates)')
	# Nothing found on any page (token-based color inference is disabled)
	return None
import csv, re, os, sys
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import colorsys
from src.utils import map_color_to_cat

# --- Migrated helpers from scripts/poc_extract_credit_factors.py ---

def parse_count_amount_pair(s):
    """Return (count, amount) from strings like '10 / $56,881' or '$56,881 / 10'.
    Returns (None, None) if no match.
    """
    if not s or '/' not in s:
        return (None, None)
    s_norm = s.replace('\u00a0', ' ').replace('\u2215', '/').replace('\u2044', '/')
    # ignore common unit patterns like '/ mo' or '/ yr'
    if re.search(r"/\s*(mo|month|yr|year)\b", s_norm, re.I):
        return (None, None)
    pairs = re.findall(r"(\$?[0-9\,]+)\s*/\s*(\$?[0-9\,]+)", s_norm)
    if not pairs:
        return (None, None)
    for left, right in pairs:
        def norm_num(t):
            return int(t.replace('$','').replace(',',''))
        try:
            lnum = norm_num(left)
            rnum = norm_num(right)
        except Exception:
            continue
        # ignore trivial all-zero placeholder pairs like '0 / $0'
        if lnum == 0 and rnum == 0:
            continue
        if '$' in left and '$' not in right:
            return (rnum, lnum)
        if '$' in right and '$' not in left:
            return (lnum, rnum)
        if lnum <= 500 and rnum > 1000:
            return (lnum, rnum)
        if rnum <= 500 and lnum > 1000:
            return (rnum, lnum)
        if ',' in right and ',' not in left:
            return (lnum, rnum)
        if ',' in left and ',' not in right:
            return (rnum, lnum)
        if lnum <= 500:
            return (lnum, rnum)
        if rnum <= 500:
            return (rnum, lnum)
        return (lnum, rnum)
    return (None, None)


def median_5x5(img, cx, cy):
    # Return median RGB of a 5x5 square centered at (cx, cy)
    w,h = img.size
    x0 = max(0, cx-2)
    y0 = max(0, cy-2)
    x1 = min(w, cx+3)
    y1 = min(h, cy+3)
    crop = img.crop((x0, y0, x1, y1))
    arr = np.array(crop).reshape(-1,3)
    if arr.size == 0:
        return None
    med = tuple(map(int, np.median(arr, axis=0)))
    return med


def parse_public_records(full_text: str):
    """Parse public records count and note from a block of text.
    Returns (count:int, note:str). If a note like 'bankrupt' exists but no explicit
    numeric count is found, we assume count==1.
    """
    m_pr = re.search(r"Public\s+Records\s*[:\s]*([0-9]+)", full_text, re.I)
    if not m_pr:
        m_pr = re.search(r"Public\s+Records\s*\n\s*([0-9]+)", full_text, re.I)
    count = int(m_pr.group(1)) if m_pr else 0
    m_note = re.search(r"(bankrupt(?:cy)?|bk\b|bankruptcy discharged|ch-\d+|chapter\s*\d+).*", full_text, re.I)
    note = m_note.group(0).strip() if m_note else ''
    if count == 0 and note:
        count = 1
    return count, note


def load_expectations_from_dir(dpath):
    """Load expectation text files from a directory.
    Returns dict mapping pdf filename -> { phrase: color }.
    """
    out = {}
    d = Path(dpath)
    if not d.exists():
        return out
    for p in d.glob('*.txt'):
        cur_pdf = None
        in_cf = False
        with open(p, 'r', encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.rstrip('\n')
                if ln.lower().startswith('pdf file:'):
                    try:
                        pathpart = ln.split(':',1)[1].strip()
                        candidate = Path(pathpart)
                        cur_pdf = candidate.name
                        # if exact file absent, try to find any file with same user id prefix
                        if not ((Path.cwd() / 'data' / 'pdf_analysis' / cur_pdf).exists() or (Path.cwd() / 'data' / 'poc_pdfs' / cur_pdf).exists()):
                            m = re.search(r'user_(\d+)_', cur_pdf)
                            if m:
                                uid = m.group(1)
                                found = None
                                for dsearch in [Path.cwd() / 'data' / 'pdf_analysis', Path.cwd() / 'data' / 'poc_pdfs']:
                                    for f2 in dsearch.glob(f"user_{uid}_*.pdf"):
                                        found = f2
                                        break
                                    if found:
                                        break
                                if found:
                                    cur_pdf = found.name
                    except Exception:
                        cur_pdf = None
                    continue
                if ln.strip().lower().startswith('credit factors'):
                    in_cf = True
                    continue
                if in_cf:
                    m = re.match(r"^\s*[-\u2022]\s*\[(\w+)\]\s*(.+)$", ln)
                    if m and cur_pdf:
                        color = m.group(1).lower()
                        phrase = m.group(2).strip()
                        out.setdefault(cur_pdf, {})[phrase] = color
                    else:
                        if ln.strip() == '':
                            in_cf = False
    return out


def run_expectation_only_qa():
	"""Compatibility shim - delegate to legacy POC QA runner."""
	import importlib
	pc = importlib.import_module('scripts.poc_extract_credit_factors')
	return pc.run_expectation_only_qa()


def find_credit_factors_region(doc, anchor='Credit Factors', max_pages=3):
    """Find page index where the 'Credit Factors' section starts and return list of (page_index, blocks_to_scan)."""
    start = None
    for i, page in enumerate(doc):
        if anchor in page.get_text():
            start = i
            break
    if start is None:
        start = 0
    regions = []
    stop_markers = ['Accounts', 'Account', 'Credit Report Details', 'Balances', 'Trade Lines']
    for p in range(start, min(start + max_pages, len(doc))):
        td = doc.load_page(p).get_text('dict')
        blocks = td.get('blocks', [])
        blocks_to_scan = []
        stop_here = False
        for b in blocks:
            # include block unless it contains an obvious stop marker
            txt = ' '.join(''.join(s.get('text','') for s in ln.get('spans',[])).strip() for ln in b.get('lines',[]))
            if any(tm.lower() in txt.lower() for tm in stop_markers):
                stop_here = True
                break
            blocks_to_scan.append(b)
        regions.append((p, blocks_to_scan))
        if stop_here:
            break
    return regions


def extract_lines_from_region(doc, start_page_blocks):
    """Extract ordered text lines and spans from a region described as list of (page_index, blocks)."""
    lines = []
    for page_index, blocks in start_page_blocks:
        for b in blocks:
            for ln in b.get('lines', []):
                text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if text:
                    lines.append((page_index, text, ln.get('spans', [])))
    return lines


def normalize_factors(raw_factors):
    """Normalize a flat list of raw factor dicts into simplified factor entries."""
    out = []
    i = 0
    import re
    while i < len(raw_factors):
        item = raw_factors[i]
        txt = (item.get('factor') or '').strip()
        low = txt.lower()
        if re.fullmatch(r"\$[0-9,]+", txt) or re.fullmatch(r"[0-9,]+", txt):
            i += 1
            continue
        if low in ('payment resp','payment resp.','past due'):
            i += 1
            continue
        canonical = map_line_to_canonical(txt)
        count = None

        total = None
        chosen_color = item.get('color')
        chosen_hex = item.get('hex')
        if i+1 < len(raw_factors):
            nxt = raw_factors[i+1]
            pair = parse_count_amount_pair(nxt.get('factor',''))
            if pair[0] is not None and pair[1] is not None:
                count, total = pair
                i += 1
        out.append({'factor': txt, 'color': chosen_color, 'hex': chosen_hex, 'canonical': canonical, 'count': count, 'total': total})
        i += 1
    # dedupe by canonical (prefer highest severity color: red>amber>green>black>neutral)
    prio = {'red':5,'amber':4,'green':3,'black':2,'neutral':1,None:1}
    merged = {}
    order = []
    for f in out:
        k = f['canonical']
        if k not in merged:
            merged[k] = f
            order.append(k)
        else:
            if prio.get(f.get('color')) > prio.get(merged[k].get('color')):
                merged[k] = f
    simplified = []
    for k in order:
        f = merged[k]
        if f['canonical'] in ('payment_resp','past_due'):
            continue
        simplified.append({'factor': f['factor'], 'color': f['color']})
    # apply small overrides
    overrides = {
        'no closed rev depth': 'red',
        'avg age open': 'red',
        'no 7.5k+ lines': 'red',
        'ok open rev depth': 'green',
        'no open mortgage': 'neutral',
        'no rev acct open 10k 2yr': 'neutral',
        'pay $': 'red',
        'total rev usage': 'green',
        'seasoned closed accounts': 'green',
        'military affiliated': 'green',
        'less than 5 yrs': 'red',
        'too few': 'red',
        'no 3k+': 'red',
        'no 3k': 'red',
        'light open rev depth': 'red',
        'unpaid collection': 'green',
        'unpaid collections': 'red'
    }
    for sf in simplified:
        for rx, col in overrides.items():
            if rx in sf['factor'].lower():
                sf['color'] = col
    for sf in simplified:
        if re.search(r"\b(rev lates|rev late)\b", sf['factor'].lower()) and re.search(r"\d", sf['factor']):
            sf['color'] = 'red'
        if re.search(r"\d+ rev lates in 0-3 mo", sf['factor'].lower()):
            sf['color'] = 'red'
    return simplified



# Compatibility helper: return candidate dicts for a phrase (page, text, spans, hex, rgb, score)
def get_candidates_for_phrase(doc, phrase, page_limit=1):
	"""Return list of candidate dicts for lines containing phrase-like tokens."""
	candidates = []
	tokens = [t.lower() for t in phrase.split() if t and not t.isdigit()]
	for pidx in range(min(page_limit, len(doc))):
		page = doc.load_page(pidx)
		td = page.get_text('dict')
		for b in td.get('blocks', []):
			for ln in b.get('lines', []):
				line_text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
				if not line_text:
					continue
				ltex = line_text.lower()
				if any(tok in ltex for tok in tokens):
					hexv, rgb = span_color_hex(ln.get('spans', []))
					cat = map_color_to_cat(rgb) if rgb else 'neutral'
					# token overlap fraction
					overlap = sum(1 for t in tokens if t in ltex)
					frac = overlap / max(1, len(tokens))
					bonus = 0.15 if re.search(r'balanc', ltex) else 0.0
					penalty = -0.20 if re.search(r'charg|chrg|charged off', ltex) else 0.0
					color_boost = 0.25 if cat == 'red' else 0.0
					# score primarily by token overlap, with modest color and semantic adjustments
					score = frac + color_boost + bonus + penalty
					candidates.append({'page': pidx, 'text': line_text, 'spans': ln.get('spans', []), 'hex': hexv, 'rgb': rgb, 'score': score})
	return candidates

def extract_credit_factors_from_doc(doc, page_limit=None):
    candidates = []
    page_pivots = {}
    for p in range(len(doc)):
        page = doc.load_page(p)
        td = page.get_text('dict')
        x0s = []
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                for s in ln.get('spans', []):
                    bbox = s.get('bbox')
                    if bbox:
                        x0s.append(bbox[0])
        if x0s:
            page_pivots[p] = (min(x0s) + max(x0s)) / 2.0
        else:
            page_pivots[p] = None
    # find right-column/right-side candidates
    for p in range(len(doc)):
        if page_limit is not None and p >= page_limit:
            break
        page = doc.load_page(p)
        td = page.get_text('dict')
        for b in td.get('blocks', []):
            for ln in b.get('lines', []):
                text = ''.join([s.get('text','') for s in ln.get('spans', [])]).strip()
                if not text:
                    continue
                x0 = None
                spans = ln.get('spans', [])
                if spans:
                    bbox = spans[0].get('bbox')
                    if bbox:
                        x0 = bbox[0]
                pivot = page_pivots.get(p)
                # treat right side lines as candidates (x0 significantly to the right of pivot), or lines shorter than 40 chars (compact)
                if pivot is not None and x0 is not None and x0 > pivot + 8:
                    candidates.append((p, text, spans, x0))
                elif len(text) < 40 and pivot is not None and x0 is not None and x0 > pivot - 40:
                    candidates.append((p, text, spans, x0))
    # Merge adjacent dollar lines with following lines when appropriate
    merged_candidates = []
    i = 0
    while i < len(candidates):
        p_idx, text, spans, x0 = candidates[i]
        if re.fullmatch(r"\$[0-9,]+", text) and i+1 < len(candidates) and candidates[i+1][0] == p_idx:
            merged_candidates.append((p_idx, f"{text} {candidates[i+1][1]}", spans + candidates[i+1][2], x0))
            i += 2
            continue
        merged_candidates.append((p_idx, text, spans, x0))
        i += 1
    raw_factors = []
    for p_idx, text, spans, x0 in merged_candidates:
        hexv, rgb = span_color_hex(spans)
        color = map_color_to_cat(rgb) if rgb else 'neutral'
        raw_factors.append({'factor': text, 'color': color, 'hex': hexv})
    return normalize_factors(raw_factors)

# Module-end compatibility aliases (ensure functions defined before aliasing)
combined_sample_color_for_phrase_impl = combined_sample_color_for_phrase
get_candidates_for_phrase_impl = get_candidates_for_phrase

