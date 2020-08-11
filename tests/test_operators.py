import operator
from typing import Any, Callable, Type, Union, cast

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from numpy import testing as npt
from pandas.core import ops

from rle_array import RLEArray, RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")

FCompareOperator = Callable[[Any, Any], Any]
FUnaryOperator = Callable[[Any], Any]
FUnaryBoolOperator = Callable[[Any], Any]
FBinaryOperator = Callable[[Any, Any], Any]
FBinaryBoolOperator = Callable[[Any, Any], Any]


@pytest.fixture
def values() -> np.ndarray:
    return np.array([2.0, 2.0, 2.0, 3.0, 3.0, 2.0, np.nan, np.nan, 1.0, 1.0])


@pytest.fixture
def scalar(values: np.ndarray) -> float:
    return float(values[0])


@pytest.fixture
def uncompressed_series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=np.arange(len(values)) + 1)


@pytest.fixture
def uncompressed_series2(values: np.ndarray) -> pd.Series:
    return pd.Series(values[::-1], index=np.arange(len(values)) + 1)


@pytest.fixture
def rle_series(values: np.ndarray) -> pd.Series:
    return pd.Series(RLEArray._from_sequence(values), index=np.arange(len(values)) + 1)


@pytest.fixture
def rle_series2(values: np.ndarray) -> pd.Series:
    return pd.Series(
        RLEArray._from_sequence(values[::-1]), index=np.arange(len(values)) + 1
    )


@pytest.fixture
def bool_values() -> np.ndarray:
    return np.array([False] * 4 + [True] * 5 + [False])


@pytest.fixture
def bool_scalar(bool_values: np.ndarray) -> bool:
    return bool(bool_values[0])


@pytest.fixture
def uncompressed_bool_series(bool_values: np.ndarray) -> pd.Series:
    return pd.Series(bool_values)


@pytest.fixture
def uncompressed_bool_series2(bool_values: np.ndarray) -> pd.Series:
    return pd.Series(bool_values[::-1])


@pytest.fixture
def rle_bool_series(bool_values: np.ndarray) -> pd.Series:
    return pd.Series(RLEArray._from_sequence(bool_values))


@pytest.fixture
def rle_bool_series2(bool_values: np.ndarray) -> pd.Series:
    # TODO: Use `index=np.arange(len(bool_values)) + 1`.
    #       For some reason, pandas casts us back to dtype=bool in that case.
    return pd.Series(RLEArray._from_sequence(bool_values[::-1]))


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
    ids=lambda op: str(op.__name__),
)
def compare_operator(request: SubRequest) -> FCompareOperator:
    return cast(FCompareOperator, request.param)


@pytest.fixture(
    params=[operator.abs, operator.neg, operator.pos], ids=lambda op: str(op.__name__)
)
def unary_operator(request: SubRequest) -> FUnaryOperator:
    return cast(FUnaryOperator, request.param)


@pytest.fixture(params=[operator.inv], ids=lambda op: str(op.__name__))
def unary_bool_operator(request: SubRequest) -> FUnaryBoolOperator:
    return cast(FUnaryBoolOperator, request.param)


@pytest.fixture(
    params=[
        operator.add,
        operator.iadd,
        ops.radd,
        operator.sub,
        operator.isub,
        ops.rsub,
        operator.mul,
        operator.imul,
        ops.rmul,
        operator.truediv,
        operator.itruediv,
        ops.rtruediv,
        operator.floordiv,
        operator.ifloordiv,
        ops.rfloordiv,
        operator.mod,
        operator.imod,
        ops.rmod,
        operator.pow,
        operator.ipow,
        ops.rpow,
    ],
    ids=lambda op: str(op.__name__),
)
def binary_operator(request: SubRequest) -> FBinaryOperator:
    return cast(FBinaryOperator, request.param)


@pytest.fixture(
    params=[
        operator.and_,
        operator.iand,
        ops.rand_,
        operator.or_,
        operator.ior,
        ops.ror_,
        operator.xor,
        operator.ixor,
        ops.rxor,
    ],
    ids=lambda op: str(op.__name__),
)
def binary_bool_operator(request: SubRequest) -> FBinaryBoolOperator:
    return cast(FBinaryBoolOperator, request.param)


def test_compare_scalar(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    scalar: float,
    compare_operator: FCompareOperator,
) -> None:
    actual = compare_operator(rle_series, scalar)
    assert actual.dtype == RLEDtype(bool)

    expected = compare_operator(uncompressed_series, scalar).astype("RLEDtype[bool]")
    pd.testing.assert_series_equal(actual, expected)


def test_compare_rle_series(
    rle_series: pd.Series,
    rle_series2: pd.Series,
    uncompressed_series: pd.Series,
    uncompressed_series2: pd.Series,
    compare_operator: FCompareOperator,
) -> None:
    actual = compare_operator(rle_series, rle_series2)
    assert actual.dtype == RLEDtype(bool)

    expected = compare_operator(uncompressed_series, uncompressed_series2).astype(
        "RLEDtype[bool]"
    )
    pd.testing.assert_series_equal(actual, expected)


def test_compare_uncompressed_series(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    compare_operator: FCompareOperator,
) -> None:
    actual = compare_operator(rle_series, uncompressed_series)
    assert actual.dtype == bool

    expected = compare_operator(uncompressed_series, uncompressed_series)
    pd.testing.assert_series_equal(actual, expected)


def test_binary_operator_scalar(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    scalar: float,
    binary_operator: FBinaryOperator,
) -> None:
    actual = binary_operator(rle_series, scalar)
    assert actual.dtype == RLEDtype(float)

    expected = binary_operator(uncompressed_series, scalar).astype("RLEDtype[float]")
    pd.testing.assert_series_equal(actual, expected)


def test_binary_operator_rle_series(
    rle_series: pd.Series,
    rle_series2: pd.Series,
    uncompressed_series: pd.Series,
    uncompressed_series2: pd.Series,
    binary_operator: FBinaryOperator,
) -> None:
    actual = binary_operator(rle_series, rle_series2)
    assert actual.dtype == RLEDtype(float)

    expected = binary_operator(uncompressed_series, uncompressed_series2).astype(
        "RLEDtype[float]"
    )
    pd.testing.assert_series_equal(actual, expected)


def test_binary_operator_uncompressed_series(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    uncompressed_series2: pd.Series,
    binary_operator: FBinaryOperator,
) -> None:
    actual = binary_operator(rle_series, uncompressed_series2)
    if getattr(binary_operator, "__name__", "???") in (
        "radd",
        "rfloordiv",
        "rmod",
        "rmul",
        "rpow",
        "rsub",
        "rtruediv",
    ):
        # pd.Series does not implement these operations correctly
        expected_dtype: Union[RLEDtype, Type[float]] = RLEDtype(float)
    else:
        expected_dtype = float

    assert actual.dtype == expected_dtype

    expected = binary_operator(uncompressed_series, uncompressed_series2).astype(
        expected_dtype
    )
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_scalar(
    rle_bool_series: pd.Series,
    uncompressed_bool_series: pd.Series,
    bool_scalar: bool,
    binary_bool_operator: FBinaryBoolOperator,
) -> None:
    actual = binary_bool_operator(rle_bool_series, bool_scalar)
    assert actual.dtype == RLEDtype(bool)

    expected = binary_bool_operator(uncompressed_bool_series, bool_scalar).astype(
        RLEDtype(bool)
    )
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_rle_series(
    rle_bool_series: pd.Series,
    rle_bool_series2: pd.Series,
    uncompressed_bool_series: pd.Series,
    uncompressed_bool_series2: pd.Series,
    binary_bool_operator: FBinaryBoolOperator,
) -> None:
    actual = binary_bool_operator(rle_bool_series, rle_bool_series2)
    assert actual.dtype == RLEDtype(bool)

    expected = binary_bool_operator(
        uncompressed_bool_series, uncompressed_bool_series2
    ).astype(RLEDtype(bool))
    pd.testing.assert_series_equal(actual, expected)


def test_binary_bool_operator_uncompressed_series(
    rle_bool_series: pd.Series,
    uncompressed_bool_series: pd.Series,
    uncompressed_bool_series2: pd.Series,
    binary_bool_operator: FBinaryBoolOperator,
) -> None:
    actual = binary_bool_operator(rle_bool_series, uncompressed_bool_series2)
    if getattr(binary_bool_operator, "__name__", "???") in ("rand_", "ror_", "rxor"):
        # pd.Series does not implement these operations correctly
        expected_dtype: Union[RLEDtype, Type[bool]] = RLEDtype(bool)
    else:
        expected_dtype = bool
    assert actual.dtype == expected_dtype

    expected = binary_bool_operator(
        uncompressed_bool_series, uncompressed_bool_series2
    ).astype(expected_dtype)
    pd.testing.assert_series_equal(actual, expected)


def test_unary_operator(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    unary_operator: FUnaryOperator,
) -> None:
    actual = unary_operator(rle_series)
    assert actual.dtype == RLEDtype(float)

    expected = unary_operator(uncompressed_series).astype(RLEDtype(float))
    pd.testing.assert_series_equal(actual, expected)


def test_unary_operator_array(
    rle_series: pd.Series,
    uncompressed_series: pd.Series,
    unary_operator: FUnaryOperator,
) -> None:
    actual = unary_operator(rle_series.array)
    assert actual.dtype == RLEDtype(float)

    expected = unary_operator(uncompressed_series.array)
    npt.assert_array_equal(actual, expected)


def test_unary_bool_operator(
    rle_bool_series: pd.Series,
    uncompressed_bool_series: pd.Series,
    unary_bool_operator: FUnaryBoolOperator,
) -> None:
    actual = unary_bool_operator(rle_bool_series)
    assert actual.dtype == RLEDtype(bool)

    expected = unary_bool_operator(uncompressed_bool_series).astype(RLEDtype(bool))
    pd.testing.assert_series_equal(actual, expected)


def test_unary_bool_operator_array(
    rle_bool_series: pd.Series,
    uncompressed_bool_series: pd.Series,
    unary_bool_operator: FUnaryBoolOperator,
) -> None:
    actual = unary_bool_operator(rle_bool_series.array)
    assert actual.dtype == RLEDtype(bool)

    expected = unary_bool_operator(uncompressed_bool_series.array)
    npt.assert_array_equal(actual, expected)


def test_different_length_raises(values: np.ndarray) -> None:
    array1 = RLEArray._from_sequence(values)
    array2 = RLEArray._from_sequence(values[:-1])
    with pytest.raises(ValueError, match="arrays have different lengths"):
        array1 + array2
