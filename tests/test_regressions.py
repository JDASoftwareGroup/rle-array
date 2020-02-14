"""
Misc collection of regression tests.
"""
import numpy as np
import numpy.testing as npt
import pytest

from rle_array.array import RLEArray

pytestmark = pytest.mark.filterwarnings("ignore:performance")


def test_object_isna():
    array = RLEArray._from_sequence(["foo", None], dtype=object)
    actual = array.isna()
    expected = np.asarray([False, True])
    npt.assert_equal(actual, expected)
