import operator

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from rle_array import RLEArray

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture
def values():
    return np.array([2.0, 2.0, 2.0, 3.0, 3.0, 2.0, np.nan, np.nan, 1.0, 1.0])


@pytest.fixture
def uncompressed_series(values):
    return pd.Series(values, index=np.arange(len(values)) + 1)


@pytest.fixture
def uncompressed_series2(values):
    return pd.Series(values[::-1], index=np.arange(len(values)) + 1)


@pytest.fixture
def rle_series(values):
    return pd.Series(RLEArray._from_sequence(values), index=np.arange(len(values)) + 1)


@pytest.fixture
def rle_series2(values):
    return pd.Series(
        RLEArray._from_sequence(values[::-1]), index=np.arange(len(values)) + 1
    )


@pytest.fixture
def bool_values():
    return np.array([False] * 5 + [True] * 4 + [False])


@pytest.fixture
def uncompressed_bool_series(bool_values):
    return pd.Series(bool_values)


@pytest.fixture
def rle_bool_series(bool_values):
    return pd.Series(RLEArray._from_sequence(bool_values))


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
    ids=lambda op: op.__name__,
)
def compare_operator(request):
    return request.param


@pytest.fixture(
    params=[operator.abs, operator.neg, operator.pos], ids=lambda op: op.__name__
)
def unary_operator(request):
    return request.param


@pytest.fixture(params=[operator.inv], ids=lambda op: op.__name__)
def unary_bool_operator(request):
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.iadd,
        operator.isub,
        operator.imul,
        operator.itruediv,
        operator.ifloordiv,
        operator.ipow,
        operator.imod,
    ],
    ids=lambda op: op.__name__,
)
def binary_operator(request):
    return request.param


@pytest.fixture(
    params=[
        operator.and_,
        operator.or_,
        operator.iand,
        operator.ior,
        operator.xor,
        operator.ixor,
    ],
    ids=lambda op: op.__name__,
)
def binary_bool_operator(request):
    return request.param


def test_compare_scalar(rle_series, uncompressed_series, compare_operator):
    actual = compare_operator(rle_series, 2.0)
    expected = compare_operator(uncompressed_series, 2.0).astype("RLEDtype[bool]")
    pd.testing.assert_series_equal(actual, expected)


def test_compare_rle_series(
    rle_series, rle_series2, uncompressed_series, uncompressed_series2, compare_operator
):
    actual = compare_operator(rle_series, rle_series2)
    expected = compare_operator(uncompressed_series, uncompressed_series2).astype(
        "RLEDtype[bool]"
    )
    pd.testing.assert_series_equal(actual, expected)


def test_compare_uncompressed_series(rle_series, uncompressed_series, compare_operator):
    actual = compare_operator(rle_series, uncompressed_series)
    expected = compare_operator(uncompressed_series, uncompressed_series)
    pd.testing.assert_series_equal(actual, expected)


def test_binary_operator_scalar(rle_series, uncompressed_series, binary_operator):
    actual = binary_operator(rle_series, 2)
    expected = binary_operator(uncompressed_series, 2)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_binary_operator_rle_series(
    rle_series, rle_series2, uncompressed_series, uncompressed_series2, binary_operator
):
    actual = binary_operator(rle_series, rle_series2)
    expected = binary_operator(uncompressed_series, uncompressed_series2)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_binary_operator_uncompressed_series(
    rle_series, uncompressed_series, binary_operator
):
    actual = binary_operator(rle_series, uncompressed_series)
    expected = binary_operator(uncompressed_series, uncompressed_series)
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_scalar(
    rle_bool_series, uncompressed_bool_series, binary_bool_operator
):
    actual = binary_bool_operator(rle_bool_series, True)
    expected = binary_bool_operator(uncompressed_bool_series, True)
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_rle_series(
    rle_bool_series, uncompressed_bool_series, binary_bool_operator
):
    actual = binary_bool_operator(rle_bool_series, rle_bool_series)
    expected = binary_bool_operator(uncompressed_bool_series, uncompressed_bool_series)
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_uncompressed_series(
    rle_bool_series, uncompressed_bool_series, binary_bool_operator
):
    actual = binary_bool_operator(rle_bool_series, uncompressed_bool_series)
    expected = binary_bool_operator(uncompressed_bool_series, uncompressed_bool_series)
    pd.testing.assert_series_equal(actual, expected)


def test_unary_operator(rle_series, uncompressed_series, unary_operator):
    actual = unary_operator(rle_series)
    expected = unary_operator(uncompressed_series)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_unary_operator_array(rle_series, uncompressed_series, unary_operator):
    actual = unary_operator(rle_series.array)
    expected = unary_operator(uncompressed_series.array)
    npt.assert_array_equal(actual, expected)


def test_unary_bool_operator(
    rle_bool_series, uncompressed_bool_series, unary_bool_operator
):
    actual = unary_bool_operator(rle_bool_series)
    expected = unary_bool_operator(uncompressed_bool_series)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_unary_bool_operator_array(
    rle_bool_series, uncompressed_bool_series, unary_bool_operator
):
    actual = unary_bool_operator(rle_bool_series.array)
    expected = unary_bool_operator(uncompressed_bool_series.array)
    npt.assert_array_equal(actual, expected)


def test_different_length_raises(values):
    array1 = RLEArray._from_sequence(values)
    array2 = RLEArray._from_sequence(values[:-1])
    with pytest.raises(ValueError, match="arrays have different lengths"):
        array1 + array2
