from typing import Any

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from pandas import testing as pdt

from rle_array import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture(
    params=[
        "single_int",
        "single_float",
        "single_float32",
        "empty_int",
        "empty_float",
        "empty_float32",
        "multi_int",
        "multi_float",
        "multi_float32",
    ]
)
def data_orig(request: SubRequest) -> pd.Series:
    if request.param == "single_int":
        return pd.Series([1], dtype=int)
    elif request.param == "single_float":
        return pd.Series([1], dtype=float)
    elif request.param == "single_float32":
        return pd.Series([1], dtype=np.float32)
    elif request.param == "empty_int":
        return pd.Series([], dtype=int)
    elif request.param == "empty_float":
        return pd.Series([], dtype=float)
    elif request.param == "empty_float32":
        return pd.Series([], dtype=np.float32)
    elif request.param == "multi_int":
        return pd.Series([1, 1, 2, 2], dtype=int)
    elif request.param == "multi_float":
        return pd.Series([1, 1, 2, 2], dtype=float)
    elif request.param == "multi_float32":
        return pd.Series([1, 1, 2, 2], dtype=np.float32)
    else:
        raise ValueError(f"Unknown data variant: {request.param}")


@pytest.fixture
def data_rle(data_orig: pd.Series) -> pd.Series:
    return data_orig.astype(RLEDtype(data_orig.dtype))


@pytest.fixture(params=[0, -1, 1, -2, 2])
def periods(request: SubRequest) -> int:
    p = request.param
    assert isinstance(p, int)
    return p


@pytest.fixture(params=[1, np.nan])
def fill_value(request: SubRequest) -> Any:
    return request.param


def test_shift(
    data_orig: pd.Series, data_rle: pd.Series, periods: int, fill_value: Any
) -> None:
    result_orig = data_orig.shift(periods=periods, fill_value=fill_value)
    result_rle = data_rle.shift(periods=periods, fill_value=fill_value)

    result_converted = result_rle.astype(result_rle.dtype._dtype)
    pdt.assert_series_equal(result_orig, result_converted)
