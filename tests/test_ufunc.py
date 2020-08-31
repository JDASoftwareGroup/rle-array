import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from rle_array import RLEArray

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture
def array_orig() -> np.ndarray:
    return np.array([1, 1, 2, 1], dtype=np.int32)


@pytest.fixture
def array_rle(array_orig: np.ndarray) -> RLEArray:
    return RLEArray._from_sequence(array_orig)


def test_square(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    expected = np.square(array_orig)
    actual = np.square(array_rle)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize("out_is_rle", [False, True])
def test_square_out(
    array_orig: np.ndarray, array_rle: RLEArray, out_is_rle: bool
) -> None:
    out_orig = np.array([0] * len(array_orig), dtype=array_orig.dtype)
    if out_is_rle:
        out_rle = RLEArray._from_sequence(out_orig)
    else:
        out_rle = out_orig.copy()

    np.square(array_orig, out=out_orig)
    np.square(array_rle, out=out_rle)

    npt.assert_array_equal(out_orig, out_rle)


def test_add_at(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    expected = np.add.at(array_orig, [0, 2], 10)
    actual = np.add.at(array_rle, [0, 2], 10)
    assert expected is None
    assert actual is None
    npt.assert_array_equal(array_orig, array_rle)


def test_divmod(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    expected1, expected2 = np.divmod(array_orig, 2)
    actual1, actual2 = np.divmod(array_rle, 2)
    npt.assert_array_equal(actual1, expected1)
    npt.assert_array_equal(actual2, expected2)


@pytest.mark.parametrize("t", [pd.Series, pd.DataFrame, pd.Index])
def test_add_unhandled(array_orig: np.ndarray, array_rle: RLEArray, t: type) -> None:
    other = t(array_orig)

    # the pandas docs say we should not handle these
    assert (
        array_rle.__array_ufunc__(np.add, "__call__", array_rle, other)
        is NotImplemented
    )


def test_2d_broadcast_add(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    # ufuncs can result in high-dimensional arrays. In that case, just return a normal NumPy array.
    other = np.vstack([array_orig, array_orig])
    assert other.shape == (2, len(array_orig))

    expected = other * array_orig
    actual = other * array_rle
    assert actual.dtype == expected.dtype
    npt.assert_array_equal(actual, expected)


def test_2d_broadcast_divmod(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    # ufuncs can result in high-dimensional arrays. In that case, just return a normal NumPy array.
    other = np.vstack([array_orig, array_orig])
    assert other.shape == (2, len(array_orig))

    expected1, expected2 = np.divmod(other, array_orig)
    actual1, actual2 = np.divmod(other, array_rle)
    assert actual1.dtype == expected1.dtype
    assert actual2.dtype == expected2.dtype
    npt.assert_array_equal(actual1, expected1)
    npt.assert_array_equal(actual2, expected2)


def test_mixed_typing_mul(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    actual = array_orig * array_rle

    expected = array_orig * array_orig
    assert actual.dtype == expected.dtype
    npt.assert_array_equal(actual, expected)


def test_mixed_typing_divmod(array_orig: np.ndarray, array_rle: RLEArray) -> None:
    actual1, actual2 = np.divmod(array_orig, array_rle)

    expected1, expected2 = np.divmod(array_orig, array_orig)
    assert actual1.dtype == expected1.dtype
    assert actual2.dtype == expected2.dtype
    npt.assert_array_equal(actual1, expected1)
    npt.assert_array_equal(actual2, expected2)
