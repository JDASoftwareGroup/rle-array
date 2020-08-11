from typing import Any, List

import numpy as np
import pytest

from rle_array import RLEDtype


@pytest.mark.parametrize(
    "a, b, expected",
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
def test_eq(a: RLEDtype, b: RLEDtype, expected: bool) -> None:
    actual = a == b
    assert actual is expected


@pytest.mark.parametrize(
    "dtype, dtypes, expected",
    [
        (  # RLE: idempotents
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8)],
            # expected
            RLEDtype(np.int8),
        ),
        (  # RLE: same types
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), RLEDtype(np.int8)],
            # expected
            RLEDtype(np.int8),
        ),
        (  # RLE: larger integer
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), RLEDtype(np.int16)],
            # expected
            RLEDtype(np.int16),
        ),
        (  # RLE: choose float
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), RLEDtype(np.float32)],
            # expected
            RLEDtype(np.float32),
        ),
        (  # RLE: use special pandas rule and chose object
            # dtype
            RLEDtype(np.bool_),
            # dtypes
            [RLEDtype(np.bool_), RLEDtype(np.float32)],
            # expected
            RLEDtype(object),
        ),
        (  # uncompressed: same types
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), np.dtype(np.int8)],
            # expected
            np.dtype(np.int8),
        ),
        (  # uncompressed: larger integer
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), np.dtype(np.int16)],
            # expected
            np.dtype(np.int16),
        ),
        (  # uncompressed: choose float
            # dtype
            RLEDtype(np.int8),
            # dtypes
            [RLEDtype(np.int8), np.dtype(np.float32)],
            # expected
            np.dtype(np.float32),
        ),
        (  # uncompressed: use special pandas rule and chose object
            # dtype
            RLEDtype(np.bool_),
            # dtypes
            [RLEDtype(np.bool_), np.dtype(np.float32)],
            # expected
            np.dtype(object),
        ),
    ],
)
def test_get_common_dtype(dtype: RLEDtype, dtypes: List[Any], expected: Any) -> None:
    actual = dtype._get_common_dtype(dtypes)
    assert actual == expected


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
def test_repr(dtype: RLEDtype, expected: str) -> None:
    assert repr(dtype) == expected
