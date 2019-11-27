import pandas as pd
import pytest

from rle_array import RLEDtype  # noqa

pytestmark = pytest.mark.filterwarnings("error:performance")


@pytest.fixture
def series():
    return pd.Series(range(10), dtype="RLEDtype[int64]")


@pytest.fixture
def df(series):
    return pd.DataFrame({"x": series, "y": series})


def test_array_slice(series):
    series.array[:]
    series.array[::-1]


def test_create_series(series):
    pass


def test_create_df(df):
    pass


def test_getitem_single(series):
    assert series[2] == 2


def test_sum(series):
    assert series.sum() == 45
