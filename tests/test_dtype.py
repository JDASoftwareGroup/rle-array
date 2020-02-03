import numpy as np
import pytest

from rle_array import RLEDtype


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            # a
            RLEDtype(int),
            # b
            RLEDtype(int),
            # expected
            True,
        ),
        (
            # a
            RLEDtype(int),
            # b
            RLEDtype(float),
            # expected
            False,
        ),
        (
            # a
            RLEDtype(int),
            # b
            RLEDtype(np.int64),
            # expected
            True,
        ),
    ],
)
def test_eq(a, b, expected):
    actual = a == b
    assert actual is expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (
            # dtype
            RLEDtype(np.dtype(int)),
            # expected
            "RLEDtype(dtype('int64'))",
        ),
        (
            # dtype
            RLEDtype(int),
            # expected
            "RLEDtype(dtype('int64'))",
        ),
    ],
)
def test_repr(dtype, expected):
    assert repr(dtype) == expected
