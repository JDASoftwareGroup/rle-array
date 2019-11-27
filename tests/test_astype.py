import numpy as np
import pandas as pd
import pytest

from rle_array import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture
def series():
    return pd.Series([1, 1, 2]).astype(RLEDtype(int))


def test_no_copy(series):
    series2 = series.astype(series.dtype, copy=False)
    assert series2.values is series.values
    assert series2.dtype == RLEDtype(int)


def test_copy_different_dtype(series):
    series2 = series.astype(RLEDtype(float), copy=False)
    assert series2.values is not series.values
    assert series2.dtype == RLEDtype(float)


def test_cast_to_np_array(series):
    series2 = series.astype(int, copy=False)
    assert series2.values is not series.values
    assert series2.dtype == np.dtype(int)
