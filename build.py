#!/usr/bin/env python3
"""
Proper build script for preauth.exe
Run from project root directory.
Embeds models and schema files so they don't need to be provided as parameters.
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
import pickle
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost

def main() -> None:
    root_dir = Path(__file__).resolve().parent
    dist_dir = root_dir / "dist"
    build_dir = root_dir / "build"

def verify_structure():
    """Verify the project has the correct structure."""
    project_root = Path.cwd()

    required_structure = {
        'src/__init__.py': 'Package marker',
        'src/predict.py': 'Main prediction module', 
        'src/ingest.py': 'Data preprocessing module',
        'src/utils.py': 'Utility functions',
        'src/column_headers.json': 'Schema configuration',
        'models/status_best.pkl': 'Trained status model',
        'models/tier_best.pkl': 'Trained tier model'
    }

    missing_files = []
    for file_path, description in required_structure.items():
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(f"{file_path} ({description})")

    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nProject structure should be:")
        print("project_root/")
        print("├── src/")
        print("│   ├── __init__.py")
        print("│   ├── predict.py")
        print("│   ├── ingest.py")
        print("│   ├── utils.py")
        print("│   └── column_headers.json")
        print("├── models/")
        print("│   ├── status_best.pkl")
        print("│   └── tier_best.pkl")
        print("└── build.py (this file)")
        return False

    return True

def clean_build():
    """Clean previous build artifacts."""
    dirs_to_clean = ['build', 'dist', '__pycache__', 'src/__pycache__']
    files_to_clean = ['*.spec']

    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"Cleaned {dir_name}")
    
    import glob
    for pattern in files_to_clean:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Cleaned {file}")

def build_executable():
    """Build the executable using PyInstaller with embedded resources."""
    
    if not verify_structure():
        sys.exit(1)
    
    print("Building preauth executable with embedded models and schema...")
    
    # Clean previous builds
    clean_build()
    
    # PyInstaller command with embedded data files
    cmd = [
        'pyinstaller',
        '--onefile',
        '--name=preauth',
        '--clean',
        '--noconfirm',
        '--console',

        '--log-level=INFO',

        # Library specifics
        '--collect-binaries=xgboost', 
        '--collect-data', 'imblearn',
        '--collect-data', 'xgboost',
        '--copy-metadata=xgboost',
        '--hidden-import=xgboost.core',
        '--hidden-import=xgboost.sklearn',
        '--hidden-import=xgboost.tracker',
        '--hidden-import=xgboost.training',

        # Your custom classes
        '--hidden-import=src.for_build',
        '--add-data=src/for_build.py:src',
        
        # Embed data files into the executable
        '--add-data=models/status_best.pkl:models',
        '--add-data=models/tier_best.pkl:models', 
        '--add-data=src/column_headers.json:src',
        
        # Hidden imports for ML libraries
        '--hidden-import=sklearn.preprocessing._label',
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors.typedefs', 
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree._utils',
        '--hidden-import=sklearn.calibration',
        '--hidden-import=sklearn.ensemble._forest',
        '--hidden-import=imblearn.pipeline',
        '--hidden-import=xgboost',
        '--hidden-import=lightgbm',
        '--hidden-import=catboost',
        '--hidden-import=imblearn',
        '--hidden-import=imblearn.over_sampling',
        '--hidden-import=imblearn.pipeline',
        
        # Exclude unnecessary modules to reduce size
        '--exclude-module=matplotlib',
        '--exclude-module=tkinter',
        '--exclude-module=PyQt5',
        '--exclude-module=PyQt6',
        '--exclude-module=PySide2',
        '--exclude-module=PySide6',
        '--exclude-module=scipy.spatial.distance',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        
        # Entry point
        'src/predict.py'
    ]

    try:

        print(f"Running PyInstaller...{cmd = }")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Build successful!")

        # Check output
        exe_name = "preauth.exe" if os.name == 'nt' else "preauth"
        exe_path = Path("dist") / exe_name

        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Created: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")

            # Quick test with embedded resources (no external files needed)
            print("\nTesting executable...")
            test_result = subprocess.run([str(exe_path), '--help'], 
                                       capture_output=True, text=True, timeout=10)
            
            if test_result.returncode == 0:
                print("✓ Help test passed!")
                
                # Test with actual prediction if test data exists
                test_file = Path("data/prefi_single_record_test.json")
                if test_file.exists():
                    print("Testing prediction with embedded models...")
                    pred_result = subprocess.run([
                        str(exe_path), 
                        '--input', str(test_file)
                    ], capture_output=True, text=True, timeout=30)
                    
                    if pred_result.returncode == 0:
                        print("✓ Prediction test passed!")
                        print("✓ Models and schema successfully embedded!")
                    else:
                        print(f"✗ Prediction test failed: {pred_result.stderr}")
                else:
                    print("ℹ No test data file found for prediction test")
                    
            else:
                print(f"✗ Help test failed: {test_result.stderr}")
        else:
            print("Warning: Executable not found")
            
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Test timed out - executable may have issues")
        sys.exit(1)
    except FileNotFoundError:
        print("PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)

def install_deps():
    """Install build dependencies."""
    deps = ['pyinstaller>=5.0']
    for dep in deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
        print(f"Installed {dep}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--install-deps':
        install_deps()
    else:
        build_executable()
