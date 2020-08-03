from typing import Callable, Union, cast

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from numpy import testing as npt
from pandas import testing as pdt

from rle_array import RLEArray, RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")

FComp = Callable[[Union[pd.Series, np.ndarray], Union[pd.Series, RLEArray]], None]


@pytest.fixture
def series_orig() -> pd.Series:
    return pd.Series([1, 1, 2, 3, 3], dtype=int)


@pytest.fixture
def array_orig(series_orig: pd.Series) -> np.ndarray:
    return series_orig.values


@pytest.fixture
def series_rle(series_orig: pd.Series) -> pd.Series:
    return series_orig.astype(RLEDtype(series_orig.dtype))


@pytest.fixture
def array_rle(series_rle: pd.Series) -> RLEArray:
    values = series_rle.values
    assert isinstance(values, RLEArray)
    return values


@pytest.fixture(params=["series", "array"])
def mode(request: SubRequest) -> str:
    m = request.param
    assert isinstance(m, str)
    return m


@pytest.fixture
def object_orig(
    series_orig: pd.Series, array_orig: np.ndarray, mode: str
) -> Union[pd.Series, np.ndarray]:
    if mode == "series":
        return series_orig
    elif mode == "array":
        return array_orig
    else:
        raise ValueError(f"Unknown mode {mode}")


@pytest.fixture
def object_rle(
    series_rle: pd.Series, array_rle: RLEArray, mode: str
) -> Union[pd.Series, RLEArray]:
    if mode == "series":
        return series_rle
    elif mode == "array":
        return array_rle
    else:
        raise ValueError(f"Unknown mode {mode}")


@pytest.fixture
def comp(mode: str) -> FComp:
    if mode == "series":
        return cast(FComp, pdt.assert_series_equal)
    elif mode == "array":
        return cast(FComp, npt.assert_array_equal)
    else:
        raise ValueError(f"Unknown mode {mode}")


def test_sum(
    object_orig: Union[pd.Series, np.ndarray],
    object_rle: Union[pd.Series, RLEArray],
    comp: FComp,
) -> None:
    elements_orig = [object_orig, object_orig]
    elements_rle = [object_rle, object_rle]
    elements_mixed = [object_rle, object_orig]

    result_orig: np.int64 = sum(elements_orig)
    result_rle: np.int64 = sum(elements_rle)
    result_mixed: np.int64 = sum(elements_mixed)

    result_converted1 = result_rle.astype(int)
    comp(result_orig, result_converted1)

    result_converted2 = result_mixed.astype(int)
    comp(result_orig, result_converted2)
