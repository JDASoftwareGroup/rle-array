"""
Misc collection of regression tests.
"""
import pickle

import numpy as np
import pytest
from numpy import testing as npt

from rle_array.array import RLEArray

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
