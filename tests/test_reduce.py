import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from rle_array import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture(params=["single", "multi", "empty", "sparse"])
def data_orig(request: SubRequest) -> pd.Series:
    if request.param == "single":
        return pd.Series([1], dtype=int)
    elif request.param == "multi":
        return pd.Series([1, 1, 2, 3, 1, 1], dtype=int)
    elif request.param == "empty":
        return pd.Series([], dtype=int)
    elif request.param == "sparse":
        return pd.Series([1, 1, np.nan, np.nan, 1, 1], dtype=float)
    else:
        raise ValueError(f"Unknown data type: {request.param}")


@pytest.fixture(params=["single", "multi", "empty"])
def data_orig_bool(request: SubRequest) -> pd.Series:
    if request.param == "single":
        return pd.Series([False], dtype=bool)
    elif request.param == "multi":
        return pd.Series([False, False, True, False], dtype=bool)
    elif request.param == "empty":
        return pd.Series([], dtype=bool)
    else:
        raise ValueError(f"Unknown data type: {request.param}")


@pytest.fixture
def data_rle(data_orig: pd.Series) -> pd.Series:
    return data_orig.astype(RLEDtype(data_orig.dtype))


@pytest.fixture
def data_rle_bool(data_orig_bool: pd.Series) -> pd.Series:
    return data_orig_bool.astype(RLEDtype(data_orig_bool.dtype))


@pytest.fixture(params=[True, False])
def skipna(request: SubRequest) -> bool:
    b = request.param
    assert isinstance(b, bool)
    return b


@pytest.fixture(
    params=["min", "max", "mean", "median", "prod", "skew", "std", "sum", "var", "kurt"]
)
def name(request: SubRequest) -> str:
    n = request.param
    assert isinstance(n, str)
    return n


@pytest.fixture(params=["any", "all"])
def name_bool(request: SubRequest) -> str:
    n = request.param
    assert isinstance(n, str)
    return n


@pytest.fixture(params=["max", "mean", "median", "min", "prod", "std", "sum", "var"])
def numpy_op(request: SubRequest) -> str:
    n = request.param
    assert isinstance(n, str)
    return n


@pytest.fixture(params=["all", "any"])
def numpy_op_bool(request: SubRequest) -> str:
    op = request.param
    assert isinstance(op, str)
    return op


@pytest.fixture(params=["mean", "std", "var"])
def numpy_op_with_dtype(request: SubRequest) -> str:
    op = request.param
    assert isinstance(op, str)
    return op


def test_reduce(
    data_orig: pd.Series, data_rle: pd.Series, skipna: bool, name: str
) -> None:
    f_orig = getattr(data_orig, name)
    f_rle = getattr(data_rle, name)
    result_orig = f_orig(skipna=skipna)
    result_rle = f_rle(skipna=skipna)
    assert (
        (np.isnan(result_orig) & np.isnan(result_rle)) | (result_orig == result_rle)
    ).all()
    # don't check type here since pandas does some magic casting from numpy to python


def test_reduce_bool(
    data_orig_bool: pd.Series, data_rle_bool: pd.Series, name_bool: str
) -> None:
    f_orig = getattr(data_orig_bool, name_bool)
    f_rle = getattr(data_rle_bool, name_bool)
    result_orig = f_orig()
    result_rle = f_rle()
    assert (result_orig == result_rle).all()
    # don't check type here since pandas does some magic casting from numpy to python


def test_array_numpy_bool_axis_notimplemented(
    data_rle_bool: pd.Series, numpy_op_bool: str
) -> None:
    f = getattr(data_rle_bool.array, numpy_op_bool)
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        f(axis=2)


def test_array_numpy_bool_out_notimplemented(
    data_rle_bool: pd.Series, numpy_op_bool: str
) -> None:
    f = getattr(data_rle_bool.array, numpy_op_bool)
    out = data_rle_bool.array.copy()
    with pytest.raises(NotImplementedError, match="out parameter is not supported."):
        f(out=out)


def test_array_reduction_not_implemented(data_rle: pd.Series) -> None:
    with pytest.raises(NotImplementedError, match="reduction foo is not implemented."):
        data_rle.array._reduce(name="foo")


def test_array_numpy_bool(
    data_orig_bool: pd.Series, data_rle_bool: pd.Series, numpy_op_bool: str
) -> None:
    f = getattr(np, numpy_op_bool)
    result_orig = f(data_rle_bool.array)
    result_rle = f(data_rle_bool.array)
    assert result_orig == result_rle
    assert type(result_orig) == type(result_rle)


def test_array_numpy(data_orig: pd.Series, data_rle: pd.Series, numpy_op: str) -> None:
    f = getattr(np, numpy_op)
    result_orig = f(data_orig.array)
    result_rle = f(data_rle.array)
    assert (pd.isna(result_orig) and pd.isna(result_rle)) or (result_orig == result_rle)
    if len(data_orig) > 0:
        assert type(result_orig) == type(result_rle)
    else:
        # pandas might use pd.NA, while we still use float, see https://github.com/pandas-dev/pandas/issues/35475
        if isinstance(result_orig, type(pd.NA)):
            assert type(result_rle) == float
        else:
            assert type(result_orig) == type(result_rle)


def test_array_numpy_axis_notimplemented(data_rle: pd.Series, numpy_op: str) -> None:
    f = getattr(data_rle.array, numpy_op)
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        f(axis=2)


def test_array_numpy_out_notimplemented(data_rle: pd.Series, numpy_op: str) -> None:
    f = getattr(data_rle.array, numpy_op)
    out = data_rle.array.copy()
    with pytest.raises(NotImplementedError, match="out parameter is not supported."):
        f(out=out)


def test_array_numpy_dtype(data_rle: pd.Series, numpy_op_with_dtype: str) -> None:
    f = getattr(np, numpy_op_with_dtype)
    with pytest.raises(NotImplementedError, match="dtype parameter is not supported."):
        f(data_rle.array, dtype=np.float16)
