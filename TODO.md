
# Get rid of scripts\poc_extract_credit_factors.py
Validation tests:
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 582 -q -x
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 692 -q -x
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 692 -q -x -s
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 705 -q -x -s
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 1140 -q -x -s
 python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 1314 -q -x -s

- [x] Audit all imports/usages (done briefly earlier) and list functions used by callers. (in-progress)
Tests
  - test_pymupdf_extractor.py
    - Imports / uses: load_expectations_from_dir, find_credit_factors_region, extract_lines_from_region, span_color_hex, map_color_to_cat, combined_sample_color_for_phrase, ROOT
    - Also references (local import): color_first_search_for_phrase, run_expectation_only_qa
  - test_vector_inspector.py
    - Imports / uses: load_expectations_from_dir, map_color_to_cat, span_color_hex, ROOT
  - test_color_phrase_sampling.py
    - Imports / uses: combined_sample_color_for_phrase, map_color_to_cat
  - test_qa_review_flag.py
    - Imports / uses: run_expectation_only_qa
  - test_public_records_parsing.py
    - Imports / uses: parse_public_records
  - (Indirect/related tests: test_span_color.py, test_numeric_parsing.py, test_1314_expectations.py — inspect if they rely on the module)
Scripts / Tools
  - run_sample_extraction.py
    - Imports / uses: extract_record_level, parse_count_amount_pair, map_line_to_canonical, span_color_hex, rgb_to_hex_tuple, map_color_to_cat, combined_sample_color_for_phrase, parse_public_records, normalize_factors
  - generate_color_training.py
    - Imports / uses: median_5x5, rgb_to_hex_tuple, map_color_to_cat
  - eval_color_extractor.py
    - Imports / uses: combined_sample_color_for_phrase, map_color_to_cat
  - validate_credit_factors.py
    - Imports / uses: extract_credit_factors_from_doc
  - dump_color_mismatches.py
    - Imports / uses: combined_sample_color_for_phrase, map_color_to_cat, median_5x5, rgb_to_hex_tuple
  - validate_hammer_small_set.py
    - Imports / uses: combined_sample_color_for_phrase, map_color_to_cat
  - eval_threshold_scan.py, eval_color_sampler.py, eval_sensitivity.py
    - Use importlib.import_module('scripts.poc_extract_credit_factors') (dynamic import; may call many attrs at runtime)
Other / Notes
  - The module defines constants and side-effect functions used by tooling (e.g., ROOT, file writing from run_expectation_only_qa, output CSV writers).
  - Some code relies on slightly different names/structures (e.g., nested account dicts vs flat counts) — tests catch these semantic differences.
  - There are many occurrences of legacy key inquiries_6mo that still appear in other scripts and test expectations (we replaced many, but some unvalidated files remain).

- [x] Move any missing helpers from poc_extract_credit_factors.py into pdf_color_extraction.py, adding/adjusting tests where needed.
  - Affected ./src/utils.py, ./src/scripts/pdf_color_extraction.py
  - Created ./tests/test_pdf_color_helpers.py
  - [ ] Compare the helpers from poc_extract_credit_factors.py (parse_count_amount_pair, median_5x5, parse_public_records, load_expectations_from_dir, find_credit_factors_region, extract_lines_from_region, normalize_factors, extract_credit_factors_from_doc) to the current extraction code in pdf_color_extraction.py. Are there superior strategies/algorithms that should be adopted in pdf_color_extraction.py?
    - the POC helpers contain reliable, structured heuristics (pivot/column detection, dollar-line merging, canonical dedupe) that are generally superior to the current ad-hoc/fallback heuristics in pdf_color_extraction.py and should be adopted as the primary credit-factor extraction path; a few targeted improvements (multi-page handling, robust OCR fallbacks, fuzzy canonicalization) would make the result even more robust.
    - GIT commit 100644 to save src/scripts/pdf_color_extraction.py before big changes, since it passes the 6 random PDF extractions.
    - Recommendations:
      - [ ] Candidate discovery (POC) vs ad-hoc lines (pdf_color_extraction). Adopt the POC pivot-based candidate discovery (i.e., extract_credit_factors_from_doc) as the primary factor extraction flow and keep the phrase/fallback as a secondary rescue path.
        - POC strategy:
          - Computes a per-page pivot (x0 median) to find right-column candidates.
          - Filters by x-position and text compactness; merges adjacent dollar lines with following label lines.This geometric approach is robust for two-column credit-summary layouts.
        - Current pdf_color_extraction:
          - Tries cf_start/contiguous-block capture or falls back to a phrase-list search (factor_phrases).
          - Good fallback, but less precise for right-column/structured pages and multi-page cases.
        Implementation of Candidate discovery:
          - First round caused failures on `python -m pytest tests/test_pdf_extraction_ground_truth.py --user_id 582 -q -x`. While we got more items (some were correct), we also got pure headings and table rows (e.g., esholds (SPAN_SAT_MIN, low-v thresholds) and prefer span colors when present; add short QA rules where fallback color overrides POC neutral.
          Add targeted tests:
          Unit tests for the three failing ground-truth cases (user_582, user_1254, user_1514) to prevent regressions.
          Run full ground-truth suite and iterate until counts/colors match expected GT.uppercase company rows like "SUNCOAST CREDIT UNIO")
          - Second round will address by:
            - Normalize and dedupe POC output: run normalize_factors() and map_line_to_canonical() on POC results before merging so keys match canonical forms.
            - Merge with smarter color pick: If both fallback and POC provide the same canonical key, prefer fallback color when it's non-neutral and POC is neutral. Otherwise use higher priority color (red > green > black > neutral).
            - Filter noisy rows: Exclude uppercase-only company lines and table/account rows (refined regex rules for lines like ^\$[0-9,]+\s+[A-Z0-9 ]+$ and tokens length heuristics).
            - Improve color heuristics: Normalize span/pixel sampling thresholds (SPAN_SAT_MIN, low-v thresholds) and prefer span colors when present; add short QA rules where fallback color overrides POC neutral.
            - Add targeted tests: Unit tests for the three failing ground-truth cases (user_582, user_1254, user_1514) to prevent regressions.
            - Run full ground-truth suite and iterate until counts/colors match expected GT.

      - [ ] Dollar-line merging & numeric parsing. Keep/standardize this merging and parsing logic (already migrated). Add more test vectors (edge locale formats, parentheses, ranges) where helpful.
          - POC: merges "$123" + "Unpaid Collections" pairs and has parse_count_amount_pair() heuristics (dollar-sign heuristic, magnitude heuristic, comma heuristic).
      - [ ] Normalization & dedupe (normalize_factors). Use normalize_factors as the canonical post-processing step and extend it with:
        - fuzzy matching (normalized token edit-distance/tokens overlap) for near-duplicates,
        - a small alias map to cover common label variants,
        - more unit tests on deduping behavior.
          - POC: canonicalizes factors, dedupes by canonical key, and chooses the highest-severity color (red>amber>green>black).
          - Current code: pdf_color_extraction had factor detection but lacked the disciplined dedupe/priority merge.
      - [ ] Color sampling & classification. Keep combined_sample_color_for_phrase and ensure that factor extraction preferentially uses span-based color tokens, then fall back to pixel-sampling (median_5x5 or global cluster heuristics) as implemented.
        - Consider tuning canonical color lists and thresholds (e.g., SPAN_SAT_MIN) via small QA runs.
        - POC helpers (median_5x5, _detect_canonical_in_pixels) + span_color_hex form a good multilevel color-sampling strategy (span-first, then pixel sampling).
        - pdf_color_extraction adds advanced routines (e.g., combined_sample_color_for_phrase, color-first hammer).
      - [ ] Multi-page & context-aware pairing. Ensure page_limit is adjustable (or None) and that merged/adjacent merging logic can span pages when reasonable (e.g., when same pivot/page context).
        - Issue found in QA: some credit factors spill to a second page or have city/state on separate lines (addresses). POC merges contiguous pages only if you instruct it to scan pages; extraction should be resilient across page boundaries.
    - [ ] The organization is poor with scripts/ inside src/. Instead create a library:
        - extract stable helpers into a package module (e.g., src/lib/pdf_extraction.py), update tests/callers, then delete the old script wrapper. (Cleaner API, higher effort) ✅




- [ ] Update callers incrementally to import from the new module directly, run unit tests after each change.
- [ ] When everything is green and callers updated, remove shim and delete old file.
- [ ] Documentation:
  - [ ] Result tracker: a report from the intensive runs with the best candidate's full metrics + feature importances. It should identify what changed in the source vs. what the resulting top candidate's results were. It needs to be maintained with every major improvement, so add it to the end of every Todo list.Ideally it has a graph showing f1 score of the top candidate on the y-axis and date on the x-axis.
  - [ ] SRS
  - [ ] validation plan
  - [ ] Developer Guide: How the code and execution is organized. How can a new person jump into making a positive contribution? What are the current challenges?
  - [ ] Architecture document containing:
    - [ ] high-level overview
    - [ ] function-by-function crawl
    - [ ] current challenges
    - [ ] future enhancement opportunities
  - [ ] Review: a line-by-line explanation of what the code does, why it is necessary, and what enhancements are needed. This is a huge file.
- [ ] Pylance error cleanup
- [ ] Maintain `docs/RESULTS_TRACKER.md` with every major improvement (append a new report and update `docs/results_history.csv`)

---

--dryrun is okay
--50 samples/classifier had issues, especially with SMOTE changing from int to float.

---

- [x] Enforce int->float conversion integrity: changing 3.0 (float) to int(3) is allowed; introducing fractional floats (e.g., 3.1) is treated as an error and will fail validation.
- [ ] Clean up output, especially progress bar and ETA.
  - [ ] What, if anything, needs done to the high-correlation errors. I presume it means that we don't need both features and should drop at least 1 of them. That should also come out in the feature importance rankings. Does it?
- [ ] Are you monitoring the cv frames to see how the f1 score is trending? What do you see, in terms of our 90% goal?
- [ ] How does the final classifier perform on the test data? Update documentation.


Here's the results of my QA:
- user_1254_credit_summary_2025-09-01_095528.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1254_credit_summary_2025-09-01_095528.pdf,$1159 Unpaid Collection(s),#ebeff4,neutral,ocr ### This is green instead of neutral
- user_1254_credit_summary_2025-09-01_095528.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1254_credit_summary_2025-09-01_095528.pdf,2 RE Lates in 0-3 mo,#ebeff4,neutral,ocr ### This is green instead of neutral
- There are many more that are missing from user_1254_credit_summary_2025-09-01_095528.pdf
    - 3 Over Limit Accnt (green)
    - Current Lates (CREDIT ACCEPTANCE CO) [green]
    - Past Due Not Late [green]
    - 4 Chrgd Off Rev Accts [red]
    - 1 Rev Late in 0-3 mo [red]
    - Avg Age Open [red]
    - No 5k+ Lines [red]
    - No Closed Rev Depth [red]
    - 3 RE Lates in 2-4 yrs [red]
    - 1 Rev Late in 2-4 yrs [red]
    - 2 inqs last 2 Months [red]
    - 2 Total Inq 0-2 Mo [red]
    - Ok Open Rev Depth [green]
    - 3+ Closed Rev Accnts [green]
    - $440 Unpaid 1 Collection [black]
    - No Open Mortgage [black]
    - No Rev Acct Open 10K 2yr [black]
    - 1 inq last 2-4 Mo [black]
    - 1 Total Inq 2-5 Mo [black]
- These should have been green instead of neutral:
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,3 Charged Off Accts,#919395,neutral,ocr
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,4 Unpaid Collection(s),#ebeff4,neutral,ocr
  - user_1514_credit_summary_2025-09-01_145557.pdf,https://shifi-app-assets.s3.us-east-2.amazonaws.com/prefi-credit-summaries/2025/09/user_1514_credit_summary_2025-09-01_145557.pdf,$42683 Unpaid Collection(s),#ebeff4,neutral,ocr
- Many missing factors for user_1514_credit_summary_2025-09-01_145557.pdf:
    - 7 Over Limit Accts [green]
    - Total Rev Usage > 90 [green]
    - Past Due Not Late [green]
    - Pay $12355 so Accts < 40 [red]
    - 8 Chrgd Off Rev Accts [red]
    - 2 Rev Lates in 4-6 mo [red]
    - 2 Rev Lates in 6-12 mo [red]
    - $3029 Unpaid Collections [red]
    - 4 Rev Lates in 2-4 yrs [red]
    - 1 RE Late in 4-6 mo [red]
    - No 5k+ Lines [red]
    - 1 Chrgd Off RE Acct [red]
    - 1 Inq Last 4 Mo [red]
    - Less than 5 yrs [red]
    - Great Closed Rev Depth [green]
    - Ok Open Rev Depth [green]
    - 3+ Closed Rev Accts [green]
    - Military Affiliated [green]
    - Seasoned Closed Accounts [green]
    - Closed Accnts Over 5k [green]
    - 8+ Rev Accnts with Balances [black]
    - In Credit Counseling [black]
    - No Open Mortgage [black]
    - 1 Rev Late in 4+ yrs [black]
    - Drop Bad Auth User (BEST BUY/CBNA, and C [black]
    - No Rev Acct Open 10K 2yr [black]
    - 1 Inq last 2-4 Mo [black]
    - 1 Inq Last 4-5 mo [black]
    - 1 Total Inq 2-4 Mo [black]

In credit_factors_poc_wide.csv:
- The credit scores are correct
- The ages are correct.
- has collections wasn't one of the requested columns, but maybe it should have been. True is correct, but we can go the next level of detail. The specific counts are:
    - 12 open and 10 closed.
    - 20 open and 1 closed.
    -  3 open and 2 closed.
- The counts are off. They should be:
    - 11 red; amber isn't a color on the list; 14 greens; 9 blacks.
    - 13 red; amber isn't a color on the list; 17 greens; 4 blacks.
    - 12 red; amber isn't a color on the list; 10 greens; 5 blacks.

For poc_images, 2 of the credit factors spilled onto a second page but you only got page 1.

credit_factors_poc_qa.json with parsed color counts summary is incorrect. The correct values are:
    - 11 red; neutral isn't a color the document; 14 greens; 9 blacks.
    - 13 red; neutral isn't a color in the document; 17 greens; 4 blacks.
    - 12 red; neutral isn't a color in the document; 10 greens; 5 blacks.

