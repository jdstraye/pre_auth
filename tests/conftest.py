import os
import sys
from pathlib import Path
import warnings
import logging

# Ensure the project root and the src package are on sys.path so imports like
# `from src.utils import ...` and `import pipeline_coordinator` both work in tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Suppress or filter noisy warnings that are expected and non-actionable in tests
# - X does not have valid feature names (sklearn) occurs when feature names differ between
#   fit and predict due to DataFrame -> ndarray conversions in tests; tests still validate behavior.
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
# - SciPy optimization Deprecation warnings (disp/iprint) are out-of-scope for us
warnings.filterwarnings("ignore", message="scipy.optimize: The `disp` and `iprint` options", category=DeprecationWarning)
# - Graphviz and other libraries may warn about positional args; silence repetitive messages in tests.
warnings.filterwarnings("ignore", message="deprecate positional args", category=DeprecationWarning)
# - Scikit-learn's estimator check warnings are expected; silence these in CI
warnings.filterwarnings("ignore", message="Skipping check_estimators_overwrite_params", category=Warning)


# Reduce debug-level logger noise during tests; keep INFO and above.
logging.getLogger().setLevel(logging.INFO)

# Add CLI options for PDF extraction tests
def pytest_addoption(parser):
    parser.addoption("--n_pdfs", action="store", default=5, type=int, help="Number of random PDFs to test.")
    parser.addoption("--pdf_dir", action="store", default="data/pdf_analysis", help="Directory containing PDFs.")
    parser.addoption("--ground_truth_dir", action="store", default="data/extracted", help="Directory for ground truth JSONs.")
    parser.addoption("--user_id", action="store", default=None, help="Specific user ID to test (e.g., 705)")
