from pathlib import Path


def test_run_sample_imports_combined_sampler():
    p = Path('scripts/run_sample_extraction.py')
    txt = p.read_text()
    assert 'combined_sample_color_for_phrase' in txt, 'run_sample_extraction.py must import combined_sample_color_for_phrase to ensure sampling is performed'
