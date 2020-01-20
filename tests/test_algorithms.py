import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import rle_array._algorithms
from rle_array._algorithms import (
    calc_lengths,
    compress,
    concat,
    decompress,
    detect_changes,
    dropna,
    extend_data,
    extend_positions,
    find_single_index,
    find_slice,
    gen_iterator,
    get_len,
    recompress,
    take,
)
from rle_array.types import POSITIONS_DTYPE


@pytest.fixture(params=[True, False], autouse=True)
def with_jit(request):
    if request.param:
        yield
    else:
        mod = rle_array._algorithms
        backup = {}
        try:
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if hasattr(obj, "py_func"):
                    orig = getattr(obj, "py_func")
                    backup[attr] = obj
                    setattr(mod, attr, orig)

            yield
        finally:
            for k, v in backup.items():
                setattr(mod, k, v)


@pytest.mark.parametrize(
    "positions,expected",
    [
        (
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # expected
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # positions
            np.array([3], dtype=POSITIONS_DTYPE),
            # expected
            np.array([3], dtype=POSITIONS_DTYPE),
        ),
        (
            # positions
            np.array([3, 10], dtype=POSITIONS_DTYPE),
            # expected
            np.array([3, 7], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_calc_lengths(positions, expected):
    actual = calc_lengths(positions)
    npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "scalars,data,positions",
    [
        (
            # scalars
            np.array([], dtype=np.int8),
            # data
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # scalars
            np.array([13], dtype=np.int8),
            # data
            np.array([13], dtype=np.int8),
            # positions
            np.array([1], dtype=POSITIONS_DTYPE),
        ),
        (
            # scalars
            np.array([13, 13, 42, 42, 42, 13], dtype=np.int8),
            # data
            np.array([13, 42, 13], dtype=np.int8),
            # positions
            np.array([2, 5, 6], dtype=POSITIONS_DTYPE),
        ),
        (
            # scalars
            np.array([13, 13, np.nan, np.nan, np.nan, 13], dtype=np.float32),
            # data
            np.array([13, np.nan, 13], dtype=np.float32),
            # positions
            np.array([2, 5, 6], dtype=POSITIONS_DTYPE),
        ),
        (
            # scalars
            pd.Series(
                [
                    pd.Timestamp("2018-01-01"),
                    pd.NaT,
                    pd.NaT,
                    pd.Timestamp("2019-01-01"),
                    pd.Timestamp("2019-01-01"),
                ],
                dtype="datetime64[ns]",
            ).values,
            # data
            pd.Series(
                [pd.Timestamp("2018-01-01"), pd.NaT, pd.Timestamp("2019-01-01")],
                dtype="datetime64[ns]",
            ).values,
            # positions
            np.array([1, 3, 5], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_compress(scalars, data, positions):
    data_actual, positions_actual = compress(scalars)
    npt.assert_array_equal(data_actual, data)
    npt.assert_array_equal(positions_actual, positions)


@pytest.mark.parametrize(
    "data_parts,positions_parts,data,positions",
    [
        (
            # data_parts
            [],
            # positions_parts
            [],
            # data
            np.array([]),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_parts
            [np.array([1, 2], dtype=np.int8)],
            # positions_parts
            [np.array([10, 20], dtype=POSITIONS_DTYPE)],
            # data
            np.array([1, 2], dtype=np.int8),
            # positions
            np.array([10, 20], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_parts
            [np.array([1, 2], dtype=np.int8), np.array([3, 2], dtype=np.int8)],
            # positions_parts
            [
                np.array([10, 20], dtype=POSITIONS_DTYPE),
                np.array([1, 10], dtype=POSITIONS_DTYPE),
            ],
            # data
            np.array([1, 2, 3, 2], dtype=np.int8),
            # positions
            np.array([10, 20, 21, 30], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_parts
            [
                np.array([1, 2], dtype=np.int8),
                np.array([3, 2], dtype=np.int8),
                np.array([4, 2], dtype=np.int8),
            ],
            # positions_parts
            [
                np.array([10, 20], dtype=POSITIONS_DTYPE),
                np.array([1, 10], dtype=POSITIONS_DTYPE),
                np.array([1, 2], dtype=POSITIONS_DTYPE),
            ],
            # data
            np.array([1, 2, 3, 2, 4, 2], dtype=np.int8),
            # positions
            np.array([10, 20, 21, 30, 31, 32], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_parts
            [np.array([], dtype=np.int8), np.array([], dtype=np.int8)],
            # positions_parts
            [np.array([], dtype=POSITIONS_DTYPE), np.array([], dtype=POSITIONS_DTYPE)],
            # data
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_parts
            [np.array([1, 2], dtype=np.int8), np.array([2, 3], dtype=np.int8)],
            # positions_parts
            [
                np.array([10, 20], dtype=POSITIONS_DTYPE),
                np.array([5, 10], dtype=POSITIONS_DTYPE),
            ],
            # data
            np.array([1, 2, 3], dtype=np.int8),
            # positions
            np.array([10, 25, 30], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_concat(data_parts, positions_parts, data, positions):
    data_actual, positions_actual = concat(data_parts, positions_parts)
    npt.assert_array_equal(data_actual, data)
    npt.assert_array_equal(positions_actual, positions)


@pytest.mark.parametrize(
    "data,positions,dtype,scalars",
    [
        (
            # dtype
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # dtype
            None,
            # scalars
            np.array([], dtype=np.int8),
        ),
        (
            # dtype
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # dtype
            np.float32,
            # scalars
            np.array([], dtype=np.float32),
        ),
        (
            # dtype
            np.array([2, np.nan, 3], dtype=np.float32),
            # positions
            np.array([1, 3, 6], dtype=POSITIONS_DTYPE),
            # dtype
            None,
            # scalars
            np.array([2, np.nan, np.nan, 3, 3, 3], dtype=np.float32),
        ),
        (
            # dtype
            np.array([2, np.nan, 3], dtype=np.float32),
            # positions
            np.array([1, 3, 6], dtype=POSITIONS_DTYPE),
            # dtype
            np.float64,
            # scalars
            np.array([2, np.nan, np.nan, 3, 3, 3], dtype=np.float64),
        ),
    ],
)
def test_decompress(data, positions, dtype, scalars):
    scalars_actual = decompress(data, positions, dtype)
    npt.assert_array_equal(scalars_actual, scalars)


@pytest.mark.parametrize(
    "scalars,changes",
    [
        (
            # scalars
            np.array([], dtype=np.float32),
            # changes
            np.array([], dtype=bool),
        ),
        (
            # scalars
            np.array([1], dtype=np.float32),
            # changes
            np.array([], dtype=bool),
        ),
        (
            # scalars
            np.array([1, 1], dtype=np.float32),
            # changes
            np.array([False], dtype=bool),
        ),
        (
            # scalars
            np.array([1, 1, 2], dtype=np.float32),
            # changes
            np.array([False, True], dtype=bool),
        ),
        (
            # scalars
            np.array([1, 1, np.nan, np.nan, 2], dtype=np.float32),
            # changes
            np.array([False, True, False, True], dtype=bool),
        ),
        (
            # scalars
            pd.Series(
                [
                    pd.Timestamp("2018-01-01"),
                    pd.NaT,
                    pd.NaT,
                    pd.Timestamp("2019-01-01"),
                    pd.Timestamp("2019-01-01"),
                ],
                dtype="datetime64[ns]",
            ).values,
            # changes
            np.array([True, False, True, False], dtype=bool),
        ),
    ],
)
def test_detect_changes(scalars, changes):
    changes_actual = detect_changes(scalars)
    npt.assert_array_equal(changes_actual, changes)


@pytest.mark.parametrize(
    "data_before,positions_before,data_after,positions_after",
    [
        (
            # data_before
            np.array([], dtype=np.float32),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([], dtype=np.float32),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.2, 3.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([1.2, 3.2], dtype=np.float32),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([np.nan], dtype=np.float32),
            # positions_before
            np.array([2], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([], dtype=np.float32),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, np.nan, 2], dtype=np.float32),
            # positions_before
            np.array([2, 5, 10], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([1, 2], dtype=np.float32),
            # positions_after
            np.array([2, 7], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_dropna(data_before, positions_before, data_after, positions_after):
    data_actual, positions_actual = dropna(data_before, positions_before)
    npt.assert_array_equal(data_actual, data_after)
    npt.assert_array_equal(positions_actual, positions_after)


@pytest.mark.parametrize(
    "data,positions,i,element",
    [
        (
            # data
            np.array([42], dtype=np.int8),
            # positions
            np.array([13], dtype=POSITIONS_DTYPE),
            # i
            0,
            # element
            42,
        ),
        (
            # data
            np.array([42], dtype=np.int8),
            # positions
            np.array([13], dtype=POSITIONS_DTYPE),
            # i
            5,
            # element
            42,
        ),
        (
            # data
            np.array([42], dtype=np.int8),
            # positions
            np.array([13], dtype=POSITIONS_DTYPE),
            # i
            12,
            # element
            42,
        ),
        (
            # data
            np.array([42, 0], dtype=np.int8),
            # positions
            np.array([5, 12], dtype=POSITIONS_DTYPE),
            # i
            4,
            # element
            42,
        ),
        (
            # data
            np.array([42, 0], dtype=np.int8),
            # positions
            np.array([5, 12], dtype=POSITIONS_DTYPE),
            # i
            5,
            # element
            0,
        ),
    ],
)
def test_find_single_index_ok(data, positions, i, element):
    element_actual = find_single_index(data, positions, i)
    assert element_actual == element


@pytest.mark.parametrize(
    "data,positions,i",
    [
        (
            # data
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # i
            0,
        ),
        (
            # data
            np.array([42], dtype=np.int8),
            # positions
            np.array([13], dtype=POSITIONS_DTYPE),
            # i
            -1,
        ),
        (
            # data
            np.array([42], dtype=np.int8),
            # positions
            np.array([13], dtype=POSITIONS_DTYPE),
            # i
            13,
        ),
    ],
)
def test_find_single_index_raise(data, positions, i):
    with pytest.raises(IndexError):
        find_single_index(data, positions, i)


@pytest.mark.parametrize(
    "data_before,positions_before,s,data_after,positions_after",
    [
        (
            # data_before
            np.array([], dtype=np.int8),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(2, None),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([11], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([13, 42], dtype=np.int8),
            # positions_before
            np.array([3, 13], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None),
            # data_after
            np.array([13, 42], dtype=np.int8),
            # positions_after
            np.array([3, 13], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([10], dtype=POSITIONS_DTYPE),
            # s
            slice(9, 10),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([1], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(None, 10),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([10], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(2, 10),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([8], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5], dtype=np.int8),
            # positions_before
            np.array([1, 5, 8, 13, 20], dtype=POSITIONS_DTYPE),
            # s
            slice(2, 10),
            # data_after
            np.array([2, 3, 4], dtype=np.int8),
            # positions_after
            np.array([3, 6, 8], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(13, None),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(None, 0),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(None, 20),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([13], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5], dtype=np.int8),
            # positions_before
            np.array([1, 5, 8, 13, 20], dtype=POSITIONS_DTYPE),
            # s
            slice(-18, -10),
            # data_after
            np.array([2, 3, 4], dtype=np.int8),
            # positions_after
            np.array([3, 6, 8], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(None, -13),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([13], dtype=POSITIONS_DTYPE),
            # s
            slice(-20, None),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([13], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5], dtype=np.int8),
            # positions_before
            np.array([1, 5, 8, 13, 20], dtype=POSITIONS_DTYPE),
            # s
            slice(1, 13),
            # data_after
            np.array([2, 3, 4], dtype=np.int8),
            # positions_after
            np.array([4, 7, 12], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5], dtype=np.int8),
            # positions_before
            np.array([1, 5, 8, 13, 20], dtype=POSITIONS_DTYPE),
            # s
            slice(2, 12),
            # data_after
            np.array([2, 3, 4], dtype=np.int8),
            # positions_after
            np.array([3, 6, 10], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5], dtype=np.int8),
            # positions_before
            np.array([1, 5, 8, 13, 20], dtype=POSITIONS_DTYPE),
            # s
            slice(4, 9),
            # data_after
            np.array([2, 3, 4], dtype=np.int8),
            # positions_after
            np.array([1, 4, 5], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([], dtype=np.int8),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 2),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([42], dtype=np.int8),
            # positions_before
            np.array([16], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 7),
            # data_after
            np.array([42], dtype=np.int8),
            # positions_after
            np.array([3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2], dtype=np.int8),
            # positions_before
            np.array([3, 9], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 3),
            # data_after
            np.array([1, 2], dtype=np.int8),
            # positions_after
            np.array([1, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3], dtype=np.int8),
            # positions_before
            np.array([2, 3, 10], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 3),
            # data_after
            np.array([1, 3], dtype=np.int8),
            # positions_after
            np.array([1, 4], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4, 5, 6, 7], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 3),
            # data_after
            np.array([1, 4, 7], dtype=np.int8),
            # positions_after
            np.array([1, 2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4], dtype=np.int8),
            # positions_before
            np.array([4, 5, 10, 11], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 3),
            # data_after
            np.array([1, 3], dtype=np.int8),
            # positions_after
            np.array([2, 4], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([], dtype=np.int8),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, -1),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2], dtype=np.int8),
            # positions_before
            np.array([3, 4], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, -1),
            # data_after
            np.array([2, 1], dtype=np.int8),
            # positions_after
            np.array([1, 4], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3], dtype=np.int8),
            # positions_before
            np.array([1, 3, 9], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, -3),
            # data_after
            np.array([3, 2], dtype=np.int8),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=POSITIONS_DTYPE),
            # s
            slice(9, 2, -3),
            # data_after
            np.array([10, 7, 4], dtype=np.int8),
            # positions_after
            np.array([1, 2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=POSITIONS_DTYPE),
            # s
            slice(9, 3, -3),
            # data_after
            np.array([10, 7], dtype=np.int8),
            # positions_after
            np.array([1, 2], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([13, -1], dtype=np.int8),
            # positions_before
            np.array([1, 2], dtype=POSITIONS_DTYPE),
            # s
            slice(None, -4),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 3, 4], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 2),
            # data_after
            np.array([1, 3], dtype=np.int8),
            # positions_after
            np.array([1, 2], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 1, 2], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4], dtype=POSITIONS_DTYPE),
            # s
            slice(None, None, 2),
            # data_after
            np.array([1], dtype=np.int8),
            # positions_after
            np.array([2], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 1, 2], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4], dtype=POSITIONS_DTYPE),
            # s
            slice(4, 0, 1),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1, 2, 1, 2], dtype=np.int8),
            # positions_before
            np.array([1, 2, 3, 4], dtype=POSITIONS_DTYPE),
            # s
            slice(0, 4, -1),
            # data_after
            np.array([], dtype=np.int8),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_find_slice(data_before, positions_before, s, data_after, positions_after):
    data_actual, positions_actual = find_slice(data_before, positions_before, s)
    npt.assert_array_equal(data_actual, data_after)
    npt.assert_array_equal(positions_actual, positions_after)


@pytest.mark.parametrize(
    "data,positions,values",
    [
        (
            # data
            np.array([], dtype=np.int8),
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # values
            [],
        ),
        (
            # data
            np.array(["foo"], dtype=object),
            # positions
            np.array([3], dtype=POSITIONS_DTYPE),
            # values
            ["foo", "foo", "foo"],
        ),
        (
            # data
            np.array([1.1, np.nan, 4.2], dtype=object),
            # positions
            np.array([1, 4, 6], dtype=POSITIONS_DTYPE),
            # values
            [1.1, np.nan, np.nan, np.nan, 4.2, 4.2],
        ),
    ],
)
def test_gen_iterator(data, positions, values):
    it = gen_iterator(data, positions)
    assert hasattr(it, "__iter__")
    values_actual = list(it)
    assert values_actual == values


@pytest.mark.parametrize(
    "positions,length",
    [
        (
            # positions
            np.array([], dtype=POSITIONS_DTYPE),
            # length
            0,
        ),
        (
            # positions
            np.array([20], dtype=POSITIONS_DTYPE),
            # length
            20,
        ),
        (
            # positions
            np.array([2, 6, 20], dtype=POSITIONS_DTYPE),
            # length
            20,
        ),
    ],
)
def test_get_len(positions, length):
    length_actual = get_len(positions)
    assert length_actual == length


@pytest.mark.parametrize(
    "data_before,positions_before,data_after,positions_after",
    [
        (
            # data_before
            np.array([], dtype=np.float32),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([], dtype=np.float32),
            # positions_after
            np.array([], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.2, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([1.2], dtype=np.float32),
            # positions_after
            np.array([3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.2, np.nan, np.nan, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3, 7, 15], dtype=POSITIONS_DTYPE),
            # data_after
            np.array([1.2, np.nan, 1.2], dtype=np.float32),
            # positions_after
            np.array([2, 7, 15], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_recompress(data_before, positions_before, data_after, positions_after):
    data_actual, positions_actual = recompress(data_before, positions_before)
    npt.assert_array_equal(data_actual, data_after)
    npt.assert_array_equal(positions_actual, positions_after)


@pytest.mark.parametrize(
    "data_before,positions_before,indices,data_after,positions_after",
    [
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2]),
            # data_after
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([-3, -2, -1]),
            # data_after
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 0, 2, 1]),
            # data_after
            np.array([1.1, 1.2, 1.1], dtype=np.float32),
            # positions_after
            np.array([2, 3, 4], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 0, 2]),
            # data_after
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2, 1.1], dtype=np.float32),
            # positions_before
            np.array([2, 3, 5], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 4]),
            # data_after
            np.array([1.1], dtype=np.float32),
            # positions_after
            np.array([2], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_take_no_fill_ok(
    data_before, positions_before, indices, data_after, positions_after
):
    data_actual, positions_actual = take(
        data=data_before,
        positions=positions_before,
        indices=indices,
        allow_fill=False,
        fill_value=None,
    )
    npt.assert_array_equal(data_actual, data_after)
    npt.assert_array_equal(positions_actual, positions_after)


@pytest.mark.parametrize(
    "data_before,positions_before,indices",
    [
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2, 3]),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2, -4]),
        ),
    ],
)
def test_take_no_fill_raise(data_before, positions_before, indices):
    with pytest.raises(IndexError):
        take(
            data=data_before,
            positions=positions_before,
            indices=indices,
            allow_fill=False,
            fill_value=None,
        )


@pytest.mark.parametrize(
    "data_before,positions_before,indices,data_after,positions_after",
    [
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2]),
            # data_after
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_after
            np.array([2, 3], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([-1, -1]),
            # data_after
            np.array([np.nan], dtype=np.float32),
            # positions_after
            np.array([2], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 0, -1, 1]),
            # data_after
            np.array([1.1, np.nan, 1.1], dtype=np.float32),
            # positions_after
            np.array([2, 3, 4], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([], dtype=np.float32),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # indices
            np.array([-1]),
            # data_after
            np.array([np.nan], dtype=np.float32),
            # positions_after
            np.array([1], dtype=POSITIONS_DTYPE),
        ),
        (
            # data_before
            np.array([], dtype=np.float32),
            # positions_before
            np.array([], dtype=POSITIONS_DTYPE),
            # indices
            np.array([-1, -1]),
            # data_after
            np.array([np.nan], dtype=np.float32),
            # positions_after
            np.array([2], dtype=POSITIONS_DTYPE),
        ),
    ],
)
def test_take_fill_ok(
    data_before, positions_before, indices, data_after, positions_after
):
    data_actual, positions_actual = take(
        data=data_before,
        positions=positions_before,
        indices=indices,
        allow_fill=True,
        fill_value=np.nan,
    )
    npt.assert_array_equal(data_actual, data_after)
    npt.assert_array_equal(positions_actual, positions_after)


@pytest.mark.parametrize(
    "data_before,positions_before,indices,e",
    [
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2, 3]),
            # e
            IndexError,
        ),
        (
            # data_before
            np.array([1.1, 1.2], dtype=np.float32),
            # positions_before
            np.array([2, 3], dtype=POSITIONS_DTYPE),
            # indices
            np.array([0, 1, 2, -2]),
            # e
            ValueError,
        ),
    ],
)
def test_take_fill_raise(data_before, positions_before, indices, e):
    with pytest.raises(e):
        take(
            data=data_before,
            positions=positions_before,
            indices=indices,
            allow_fill=True,
            fill_value=np.nan,
        )


@pytest.mark.parametrize(
    "indices,allow_fill",
    [
        (
            # indices
            np.array([0]),
            # allow_fill
            False,
        ),
        (
            # indices
            np.array([-1]),
            # allow_fill
            False,
        ),
        (
            # indices
            np.array([0]),
            # allow_fill
            True,
        ),
        (
            # indices
            np.array([-2]),
            # allow_fill
            True,
        ),
    ],
)
def test_take_raises_non_empty_take(indices, allow_fill):
    data = np.array([], dtype=POSITIONS_DTYPE)
    positions = np.array([], dtype=np.float32)
    with pytest.raises(IndexError, match="cannot do a non-empty take"):
        take(
            data=data,
            positions=positions,
            indices=indices,
            allow_fill=allow_fill,
            fill_value=np.nan,
        )


@pytest.mark.parametrize(
    "positions1, positions2, expected",
    [
        (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)),
        (np.array([1, 5]), np.array([], dtype=int), np.array([1, 5])),
        (np.array([], dtype=int), np.array([1, 5]), np.array([1, 5])),
        (np.array([2, 7]), np.array([1, 5, 7]), np.array([1, 2, 5, 7])),
    ],
)
def test_extend_positions(positions1, positions2, expected):
    actual = extend_positions(positions1, positions2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "data, positions, extended_positions, expected",
    [
        (np.array([]), np.array([], dtype=int), np.array([], dtype=int), np.array([])),
        (
            np.array([2.0]),
            np.array([10]),
            np.array([3, 7, 10]),
            np.array([2.0, 2.0, 2.0]),
        ),
        (
            np.array([3.0, 2.0, 8.0]),
            np.array([3, 7, 10]),
            np.array([2, 3, 4, 6, 7, 8, 10]),
            np.array([3.0, 3.0, 2.0, 2.0, 2.0, 8.0, 8.0]),
        ),
    ],
)
def test_extend_data(data, positions, extended_positions, expected):
    actual = extend_data(data, positions, extended_positions)
    np.testing.assert_array_equal(actual, expected)
