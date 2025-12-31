import pandas as pd
import numpy as np
import pytest

from src.debug_library import DataValidator


def test_compare_frames_allows_int_to_float_warning(tmp_path):
    dv = DataValidator(name="test")
    before = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64"), "b": pd.Series([0, 1, 1], dtype="int64")})
    after = pd.DataFrame({"a": pd.Series([1.0, 2.0, 3.0], dtype="float64"), "b": pd.Series([0.0, 1.0, 1.0], dtype="float64")})

    # Should not raise for int->float conversions
    report = dv.compare_frames(before, after, operation="int_to_float_test")
    assert report["dtype_changes"]
    # dtype change should be a warning, not an error
    assert not report["errors"]
    assert report["passed"]


def test_compare_frames_int_to_object_is_error(tmp_path):
    dv = DataValidator(name="test")
    before = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    after = pd.DataFrame({"a": pd.Series(["x", "y", "z"], dtype="object")})

    with pytest.raises(ValueError):
        dv.compare_frames(before, after, operation="int_to_object_test")


def test_compare_frames_int_to_float_fractional_is_error(tmp_path):
    dv = DataValidator(name="test")
    before = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    after = pd.DataFrame({"a": pd.Series([1.0, 2.5, 3.0], dtype="float64")})

    with pytest.raises(ValueError):
        dv.compare_frames(before, after, operation="int_to_fractional_test")


def test_compare_frames_int_to_float_integer_equiv_is_warning(tmp_path):
    dv = DataValidator(name="test")
    before = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    after = pd.DataFrame({"a": pd.Series([1.0, 2.0, 3.0], dtype="float64")})

    report = dv.compare_frames(before, after, operation="int_to_float_integer_equiv_test")
    assert report["dtype_changes"]
    assert not report["errors"]
    assert report["passed"]
