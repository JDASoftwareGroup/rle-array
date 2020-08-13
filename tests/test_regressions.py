"""
Misc collection of regression tests.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas.core.dtypes.common import ensure_int_or_float

from rle_array import RLEArray, RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


def test_object_isna() -> None:
    array = RLEArray._from_sequence(["foo", None], dtype=object)
    actual = array.isna()
    expected = np.asarray([False, True])
    npt.assert_equal(actual, expected)


def test_mean_divisor_overflow() -> None:
    # https://github.com/JDASoftwareGroup/rle-array/issues/22
    array = RLEArray._from_sequence([1] * 256, dtype=np.uint8)
    assert array.mean() == 1


def test_pickle() -> None:
    array = RLEArray._from_sequence([1])

    # roundtrip
    s = pickle.dumps(array)
    array2 = pickle.loads(s)
    npt.assert_array_equal(array, array2)

    # views must not be linked (A)
    array2_orig = array2.copy()
    array[:] = 2
    npt.assert_array_equal(array2, array2_orig)

    # views must not be linked (B)
    array_orig = array.copy()
    array2[:] = 3
    npt.assert_array_equal(array, array_orig)


def test_inplace_update() -> None:
    array = RLEArray._from_sequence([1], dtype=np.int64)
    array[[True]] = 2

    expected = np.array([2], dtype=np.int64)
    npt.assert_array_equal(array, expected)

    assert array._dtype._dtype == np.int64
    assert array._data.dtype == np.int64


def test_append_mixed() -> None:
    actual = pd.concat(
        [pd.Series([1], dtype=np.int8), pd.Series([1], dtype=RLEDtype(np.int8))]
    )
    assert actual.dtype == np.int8


def test_bool_ensure_int_or_float() -> None:
    array = RLEArray._from_sequence([False, True], dtype=np.bool_)
    actual = ensure_int_or_float(array)

    expected = np.array([0, 1], dtype=np.int64)
    assert actual.dtype == expected.dtype
    npt.assert_array_equal(actual, expected)


def test_groupby_bool_first() -> None:
    df = pd.DataFrame({"x": pd.Series([True, True], dtype=RLEDtype(bool)), "g": 1})
    series = df.groupby("g")["x"].first()
    assert series.dtype == RLEDtype(bool)

    expected = RLEArray._from_sequence([True])
    npt.assert_array_equal(series.array, expected)


def test_from_sequence_bool() -> None:
    array = RLEArray._from_sequence(
        np.array([0, 1], dtype=np.int64), dtype=RLEDtype(bool)
    )
    npt.assert_array_equal(array, np.array([False, True]))

    array = RLEArray._from_sequence(
        np.array([0.0, 1.0], dtype=np.float64), dtype=RLEDtype(bool)
    )
    npt.assert_array_equal(array, np.array([False, True]))

    with pytest.raises(TypeError, match="Need to pass bool-like values"):
        RLEArray._from_sequence(np.array([1, 2], dtype=np.int64), dtype=RLEDtype(bool))

    with pytest.raises(TypeError, match="Need to pass bool-like values"):
        RLEArray._from_sequence(np.array([-1, 1], dtype=np.int64), dtype=RLEDtype(bool))

    with pytest.raises(TypeError, match="Masked booleans are not supported"):
        RLEArray._from_sequence(
            np.array([np.nan, 1.0], dtype=np.float64), dtype=RLEDtype(bool)
        )


def test_groupby_bool_sum() -> None:
    # Cython routines for integer addition are not available, so we need to accept floats here.
    df = pd.DataFrame({"x": pd.Series([True, True], dtype=RLEDtype(bool)), "g": 1})
    series = df.groupby("g")["x"].sum()
    assert series.dtype == np.float64

    expected = np.array([2], dtype=np.float64)
    npt.assert_array_equal(series.to_numpy(), expected)


def test_factorize_int() -> None:
    array = RLEArray._from_sequence([42, -10, -10], dtype=RLEDtype(np.int32))
    codes_actual, uniques_actual = array.factorize()

    codes_expected = np.array([0, 1, 1], dtype=np.int64)
    assert codes_actual.dtype == codes_expected.dtype
    npt.assert_array_equal(codes_actual, codes_expected)

    uniques_expected = RLEArray._from_sequence([42, -10], dtype=np.int32)
    assert uniques_actual.dtype == uniques_expected.dtype
    npt.assert_array_equal(uniques_actual, uniques_expected)
