import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from rle_array import RLEArray
from rle_array._algorithms import compress

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture
def array_orig():
    return np.array([1, 1, 2, 1], dtype=np.int32)


@pytest.fixture
def array_rle(array_orig):
    return RLEArray(*compress(array_orig))


def test_square(array_orig, array_rle):
    expected = np.square(array_orig)
    actual = np.square(array_rle)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("out_is_rle", [False, True])
def test_square_out(array_orig, array_rle, out_is_rle):
    out_orig = np.array([0] * len(array_orig), dtype=array_orig.dtype)
    if out_is_rle:
        out_rle = RLEArray(*compress(out_orig))
    else:
        out_rle = out_orig.copy()

    np.square(array_orig, out=out_orig)
    np.square(array_rle, out=out_rle)

    npt.assert_array_equal(out_orig, out_rle)


def test_add_at(array_orig, array_rle):
    expected = np.add.at(array_orig, [0, 2], 10)
    actual = np.add.at(array_rle, [0, 2], 10)
    assert expected is None
    assert actual is None
    npt.assert_array_equal(array_orig, array_rle)


def test_divmod(array_orig, array_rle):
    expected1, expected2 = np.divmod(array_orig, 2)
    actual1, actual2 = np.divmod(array_rle, 2)
    npt.assert_array_equal(actual1, expected1)
    npt.assert_array_equal(actual2, expected2)


@pytest.mark.parametrize("t", [pd.Series, pd.DataFrame, pd.Index])
def test_add_unhandled(array_orig, array_rle, t):
    other = t(array_orig)

    # the pandas docs say we should not handle these
    assert (
        array_rle.__array_ufunc__(np.add, "__call__", array_rle, other)
        is NotImplemented
    )
