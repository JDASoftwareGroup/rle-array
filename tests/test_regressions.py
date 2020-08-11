"""
Misc collection of regression tests.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

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
