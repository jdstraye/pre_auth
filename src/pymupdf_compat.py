"""Compatibility shim for importing PyMuPDF (fitz) while addressing Swig/PyMuPDF
DeprecationWarnings about builtin Swig types missing __module__ attributes.

This module imports fitz while briefly suppressing the specific DeprecationWarning
and then sets the __module__ attribute on common Swig types (when present) to
prevent those warnings from recurring. Code in this repo should import fitz
from here (``from src.pymupdf_compat import fitz``) rather than importing the
upstream module directly.
"""
import warnings

def _import_and_patch_fitz():
    # suppress the specific deprecation messages during import
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"builtin type .* has no __module__ attribute",
            category=DeprecationWarning,
        )
        import fitz as _fitz

    # Patch common Swig types to include __module__ attribute if missing.
    # This prevents other code from re-triggering DeprecationWarnings later.
    try:
        for name in ("SwigPyPacked", "SwigPyObject", "swigvarlink"):
            try:
                t = getattr(_fitz, name, None)
                if t is not None and getattr(t, "__module__", None) is None:
                    try:
                        t.__module__ = _fitz.__name__
                    except Exception:
                        # best-effort; ignore if attribute setting fails
                        pass
            except Exception:
                # ignore any attribute errors
                pass
    except Exception:
        pass
    return _fitz

# Publicly export a ready-to-use fitz object
fitz = _import_and_patch_fitz()
