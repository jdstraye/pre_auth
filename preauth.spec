# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/predict.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('src/column_headers.json', 'src'), ('models/status_best.pkl', 'models'), ('models/tier_best.pkl', 'models'), ('src/column_headers.json', 'src')],
    hiddenimports=['sklearn.preprocessing._label', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'xgboost', 'lightgbm', 'catboost', 'imblearn', 'imblearn.over_sampling', 'imblearn.pipeline'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='preauth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
