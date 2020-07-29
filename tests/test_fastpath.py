import pandas as pd
import pytest

from rle_array import RLEDtype  # noqa

pytestmark = pytest.mark.filterwarnings("error:performance")


@pytest.fixture
def series() -> pd.Series:
    return pd.Series(range(10), dtype="RLEDtype[int64]")


@pytest.fixture
def df(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"x": series, "y": series})


def test_array_slice(series: pd.Series) -> None:
    series.array[:]
    series.array[::-1]


def test_create_series(series: pd.Series) -> None:
    pass


def test_create_df(df: pd.Series) -> None:
    pass


def test_getitem_single(series: pd.Series) -> None:
    assert series[2] == 2


def test_sum(series: pd.Series) -> None:
    assert series.sum() == 45
