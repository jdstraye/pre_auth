# Validate HAMMER on a Small Set of PDFs

# List of sample PDFs and target phrase for validation
sample_pdfs = [
    'data/pdf_analysis/user_733_credit_summary_2025-09-01_105309.pdf',
    'data/pdf_analysis/user_708_credit_summary_2025-09-01_105127.pdf',
    'data/pdf_analysis/user_813_credit_summary_2025-09-01_091838.pdf',
    'data/pdf_analysis/user_651_credit_summary_2025-09-01_105006.pdf',
    'data/pdf_analysis/user_1289_credit_summary_2025-09-01_102929.pdf',
]
target_phrase = '1 Inq Last 4 Mo'

from src.scripts.pdf_color_extraction import combined_sample_color_for_phrase, map_color_to_cat
import fitz

print('PDF, Detected, Color, Method')
for pdf in sample_pdfs:
    try:
        doc = fitz.open(pdf)
        res = combined_sample_color_for_phrase(doc, target_phrase, expected_color=None, page_limit=1)
        if res:
            _, text, _, rgb, _, method = res
            cat = map_color_to_cat(rgb) if rgb is not None else 'neutral'
            print(f'{pdf}, {cat}, {rgb}, {method}')
        else:
            print(f'{pdf}, NOT FOUND, -, -')
    except Exception as e:
        print(f'{pdf}, ERROR, -, -')
