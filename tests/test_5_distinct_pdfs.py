import json
import sys
import argparse
from src.pymupdf_compat import fitz
from pathlib import Path
from datetime import datetime
import pytest

# Ensure we can import from the scripts directory
sys.path.insert(0, str(Path.cwd()))

from src.scripts.pdf_color_extraction import combined_sample_color_for_phrase
from src.utils import map_color_to_cat

# Canonical ground truth for 5-PDF validation
GROUND_TRUTH = {
    'user_426_credit_summary_2025-09-01_095930.pdf': {
        '1 Rev Late in 0-3 mo': 'red',
        '2 RE Lates in 6-12 mo': 'red',
        'No Closed Rev Depth': 'red',
        'Avg Age Open': 'red',
        'No 7.5k+ Lines': 'red',
        '4 RE Lates in 4-6 mo': 'green',
        'Past Due Not Late': 'green',
        'Ok Open Rev Depth': 'green',
        'Credit Score': 'black',
        'Report Date': 'black',
        'Open Accounts': 'black',
    },
    'user_577_credit_summary_2025-09-01_104651.pdf': {
        'Pay $4976 so Accts < 40%': 'red',
        '1 Rev Late in 6-12 mo': 'red',
        'No Closed Rev Depth': 'red',
        'No 5k+ Lines': 'red',
        'Less than 5 yrs': 'red',
        'Total Rev Usage > 55%': 'green',
        'Seasoned Closed Accounts': 'green',
        'Ok Open Rev Depth': 'green',
        '3+ Closed Rev Accnts': 'green',
        '6+ Closed RE Accounts': 'green',
        'Paid Off 200k+ RE/RE': 'green',
        'Credit Score': 'black',
        'Open Accounts': 'black',
    },
    'user_599_credit_summary_2025-09-01_104729.pdf': {
        'No Closed Rev Depth': 'red',
        'Less than 5 yrs': 'red',
        '7k+ line for 4+yrs': 'green',
        'Ok Open Rev Depth': 'green',
        '1+ Closed Rev Accnts': 'green',
        'No Open Mortgage': 'black',
        'Seasoned Closed Accounts': 'black',
        'Credit Score': 'black',
        'Open Accounts': 'black',
    },
    'user_1300_credit_summary_2025-09-01_132946.pdf': {
        'Pay $17467 so Accts < 40%': 'red',
        'Total Rev Usage > 55%': 'green',
        '7k+ line for 12+yrs': 'green',
        'Ok Open Rev Depth': 'green',
        '3+ Closed Rev Accnts': 'green',
        'Ok Closed Rev Depth': 'green',
        'Military Affiliated': 'green',
        'Seasoned Closed Accounts': 'green',
        'Credit Score': 'black',
        'No Open Mortgage': 'black',
    },
    'user_1314_credit_summary_2025-09-01_092724.pdf': {
        'Pay $12355 so Accts < 40%': 'red',
        '8 Chrgd Off Rev Accts': 'red',
        '2 Rev Lates in 4-6 mo': 'red',
        '2 Rev Lates in 6-12 mo': 'red',
        '$3029 Unpaid Collections': 'red',
        '4 Rev Lates in 2-4 yrs': 'red',
        '1 RE Late in 4-6 mo': 'red',
        'No 5k+ Lines': 'red',
        '1 Chrgd Off RE Acct': 'red',
        '1 Inq Last 4 Mo': 'red',
        'Less than 5 yrs': 'red',
        '6 Charged Off Accts': 'green',
        '4 Unpaid Collection(s)': 'green',
        '$16320 Unpaid Collection(s)': 'green',
        '7 Over Limit Accnt': 'green',
        'Total Rev Usage > 90%': 'green',
        'Past Due Not Late': 'green',
        'Great Closed Rev Depth': 'green',
        'Ok Open Rev Depth': 'green',
        '3+ Closed Rev Accnts': 'green',
        'Military Affiliated': 'green',
        'Seasoned Closed Accounts': 'green',
        'Closed Accnts Over 5k': 'green',
        'Credit Score': 'black',
        'Report Date': 'black',
    }
}

def find_pdfs(search_dirs=None):
    """Auto-detect PDF locations for the ground truth set."""
    if search_dirs is None:
        search_dirs = [
            Path.cwd(),
            Path.cwd() / "pdfs",
            Path.cwd() / "test_pdfs",
            Path.cwd() / "data",
            Path.cwd() / "data/pdf_analysis",
            Path.home() / "Downloads"
        ]
    found_pdfs = {}
    for pdf_name in GROUND_TRUTH.keys():
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            matches = list(search_dir.glob(f"**/{pdf_name}"))
            if matches:
                found_pdfs[pdf_name] = matches[0]
                break
    return found_pdfs

class PDFColorValidator:
    def __init__(self, log_dir="validation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

    def test_pdf(self, pdf_path, ground_truth):
        pdf_name = Path(pdf_path).name
        doc = fitz.open(pdf_path)
        pdf_results = {
            'pdf': pdf_name,
            'timestamp': datetime.now().isoformat(),
            'phrases': {},
            'summary': {'total': 0, 'correct': 0, 'incorrect': 0}
        }
        
        print(f"\n🔍 Validating {pdf_name}...")
        for phrase, expected_color in ground_truth.items():
            res = combined_sample_color_for_phrase(doc, phrase, page_limit=3)
            if res is None:
                result = {'color': 'not_found', 'text': '', 'hex': None, 'page': None, 'found': False}
            else:
                pidx, line_text, hexv, rgb, bbox, method = res
                color = map_color_to_cat(rgb) if rgb is not None else 'not_found'
                result = {
                    'color': color,
                    'text': line_text,
                    'hex': hexv,
                    'page': pidx,
                    'found': True
                }
            
            actual_color = result.get('color', 'not_found')
            is_correct = (expected_color == actual_color)
            pdf_results['phrases'][phrase] = {**result, 'expected': expected_color, 'match': is_correct}
            
            pdf_results['summary']['total'] += 1
            if is_correct:
                pdf_results['summary']['correct'] += 1
                print(f"  ✅ '{phrase}': {actual_color}")
            else:
                pdf_results['summary']['incorrect'] += 1
                print(f"  ❌ '{phrase}': Expected {expected_color}, Got {actual_color}")

        self.results.append(pdf_results)
        self._save_incremental(pdf_results)
        return pdf_results

    def _save_incremental(self, pdf_result):
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(pdf_result) + '\n')

    def generate_report(self):
        total_phrases = sum(r['summary']['total'] for r in self.results)
        total_correct = sum(r['summary']['correct'] for r in self.results)
        accuracy = total_correct / total_phrases if total_phrases > 0 else 0
        
        report = {
            'overall_accuracy': accuracy,
            'total_correct': total_correct,
            'total_incorrect': total_phrases - total_correct,
            'by_pdf': []
        }
        for r in self.results:
            report['by_pdf'].append({
                'pdf': r['pdf'],
                'accuracy': r['summary']['correct'] / r['summary']['total'],
                'correct': r['summary']['correct'],
                'total': r['summary']['total']
            })
        return report

@pytest.mark.timeout(120)
def test_full_5_pdf_validation():
    """Pytest entry point for the 5-PDF validation suite."""
    found_pdfs = find_pdfs()
    missing = set(GROUND_TRUTH.keys()) - set(found_pdfs.keys())
    assert not missing, f"Missing PDFs required for test: {missing}"
    
    validator = PDFColorValidator(log_dir="pytest_validation_logs")
    for pdf_name, pdf_path in found_pdfs.items():
        validator.test_pdf(str(pdf_path), GROUND_TRUTH[pdf_name])
    
    report = validator.generate_report()
    
    # Assertions based on your criteria
    assert report['overall_accuracy'] >= 0.80, f"Overall accuracy {report['overall_accuracy']:.1%} below 80%"
    
    u426 = next(r for r in report['by_pdf'] if 'user_426' in r['pdf'])
    assert u426['accuracy'] == 1.0, f"user_426 accuracy {u426['accuracy']:.1%} below 100%"
    
    u599 = next(r for r in report['by_pdf'] if 'user_599' in r['pdf'])
    assert u599['accuracy'] == 1.0, f"user_599 accuracy {u599['accuracy']:.1%} below 100%"
    
    u1314 = next(r for r in report['by_pdf'] if 'user_1314' in r['pdf'])
    assert u1314['accuracy'] >= 0.80, f"user_1314 accuracy {u1314['accuracy']:.1%} below 80%"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDFs')
    args = parser.parse_args()
    
    print("🚀 Starting PDF Color Validation...")
    pdfs = find_pdfs([Path(args.pdf_dir)] if args.pdf_dir else None)
    
    if len(pdfs) < len(GROUND_TRUTH):
        print(f"⚠️  Only found {len(pdfs)}/{len(GROUND_TRUTH)} PDFs. Check your paths.")
        sys.exit(1)
        
    val = PDFColorValidator()
    for name, path in pdfs.items():
        val.test_pdf(str(path), GROUND_TRUTH[name])
    
    rep = val.generate_report()
    print(f"\n📊 FINAL ACCURACY: {rep['overall_accuracy']:.1%}")