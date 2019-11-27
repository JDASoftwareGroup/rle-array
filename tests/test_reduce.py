import numpy as np
import pandas as pd
import pytest

from rle_array import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture(params=["single", "multi", "empty", "sparse"])
def data_orig(request):
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
def data_orig_bool(request):
    if request.param == "single":
        return pd.Series([False], dtype=bool)
    elif request.param == "multi":
        return pd.Series([False, False, True, False], dtype=bool)
    elif request.param == "empty":
        return pd.Series([], dtype=bool)
    else:
        raise ValueError(f"Unknown data type: {request.param}")


@pytest.fixture
def data_rle(data_orig):
    return data_orig.astype(RLEDtype(data_orig.dtype))


@pytest.fixture
def data_rle_bool(data_orig_bool):
    return data_orig_bool.astype(RLEDtype(data_orig_bool.dtype))


@pytest.fixture(params=[True, False])
def skipna(request):
    return request.param


@pytest.fixture(
    params=["min", "max", "mean", "median", "prod", "skew", "std", "sum", "var", "kurt"]
)
def name(request):
    return request.param


@pytest.fixture(params=["any", "all"])
def name_bool(request):
    return request.param


@pytest.fixture(params=["max", "mean", "median", "min", "prod", "std", "sum", "var"])
def numpy_op(request):
    return request.param


@pytest.fixture(params=["all", "any"])
def numpy_op_bool(request):
    return request.param


@pytest.fixture(params=["mean", "std", "var"])
def numpy_op_with_dtype(request):
    return request.param


def test_reduce(data_orig, data_rle, skipna, name):
    f_orig = getattr(data_orig, name)
    f_rle = getattr(data_rle, name)
    result_orig = f_orig(skipna=skipna)
    result_rle = f_rle(skipna=skipna)
    assert (
        (np.isnan(result_orig) & np.isnan(result_rle)) | (result_orig == result_rle)
    ).all()
    # don't check type here since pandas does some magic casting from numpy to python


def test_reduce_bool(data_orig_bool, data_rle_bool, name_bool):
    f_orig = getattr(data_orig_bool, name_bool)
    f_rle = getattr(data_rle_bool, name_bool)
    result_orig = f_orig()
    result_rle = f_rle()
    assert (result_orig == result_rle).all()
    # don't check type here since pandas does some magic casting from numpy to python


def test_array_numpy_bool_axis_notimplemented(data_rle_bool, numpy_op_bool):
    f = getattr(data_rle_bool.array, numpy_op_bool)
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        f(axis=2)


def test_array_numpy_bool_out_notimplemented(data_rle_bool, numpy_op_bool):
    f = getattr(data_rle_bool.array, numpy_op_bool)
    out = data_rle_bool.array.copy()
    with pytest.raises(NotImplementedError, match="out parameter is not supported."):
        f(out=out)


def test_array_reduction_not_implemented(data_rle):
    with pytest.raises(NotImplementedError, match="reduction foo is not implemented."):
        data_rle.array._reduce(name="foo")


def test_array_numpy_bool(data_orig_bool, data_rle_bool, numpy_op_bool):
    f = getattr(np, numpy_op_bool)
    result_orig = f(data_rle_bool.array)
    result_rle = f(data_rle_bool.array)
    assert result_orig == result_rle
    assert type(result_orig) == type(result_rle)


def test_array_numpy(data_orig, data_rle, numpy_op):
    f = getattr(np, numpy_op)
    result_orig = f(data_orig.array)
    result_rle = f(data_rle.array)
    assert (np.isnan(result_orig) and np.isnan(result_rle)) or (
        result_orig == result_rle
    )
    assert type(result_orig) == type(result_rle)


def test_array_numpy_axis_notimplemented(data_rle, numpy_op):
    f = getattr(data_rle.array, numpy_op)
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        f(axis=2)


def test_array_numpy_out_notimplemented(data_rle, numpy_op):
    f = getattr(data_rle.array, numpy_op)
    out = data_rle.array.copy()
    with pytest.raises(NotImplementedError, match="out parameter is not supported."):
        f(out=out)


def test_array_numpy_dtype(data_rle, numpy_op_with_dtype):
    f = getattr(np, numpy_op_with_dtype)
    with pytest.raises(NotImplementedError, match="dtype parameter is not supported."):
        f(data_rle.array, dtype=np.float16)
