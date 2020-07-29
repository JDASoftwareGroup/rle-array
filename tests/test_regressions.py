"""
Misc collection of regression tests.
"""
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
