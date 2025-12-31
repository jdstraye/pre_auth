from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / 'data' / 'color_training'


def load_phrases(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def test_user_954_phrases():
    p = TRAIN_DIR / 'user_954_credit_summary_2025-09-01_102100.p1.right.phrases.txt'
    assert p.exists()
    lines = load_phrases(p)
    # find key phrases
    by_phrase = {l['phrase']: l for l in lines}
    assert 'Too Few Open Rev Accounts' in by_phrase
    assert by_phrase['Too Few Open Rev Accounts']['cat'] == 'red'
    assert '7k+ line for 12+yrs' in by_phrase
    assert by_phrase['7k+ line for 12+yrs']['cat'] == 'green'


def test_user_692_phrases():
    p = TRAIN_DIR / 'user_692_credit_summary_2025-09-01_105038.p1.right.phrases.txt'
    assert p.exists()
    lines = load_phrases(p)
    by_phrase = {l['phrase']: l for l in lines}
    assert '40 RE Lates in 2-4 yrs' in by_phrase
    assert by_phrase['40 RE Lates in 2-4 yrs']['cat'] == 'red'
    assert 'Less than 5 yrs' in by_phrase
    assert by_phrase['Less than 5 yrs']['cat'] == 'red'
