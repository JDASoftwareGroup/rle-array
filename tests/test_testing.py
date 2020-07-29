import itertools
from typing import List, cast

import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from pandas import testing as pdt

from rle_array.testing import (
    const_col,
    dim_col,
    generate_example,
    generate_test_dataframe,
)


@pytest.mark.parametrize(
    "dims, expected",
    [
        (
            # dims
            [1],
            # expected
            "const_1",
        ),
        (
            # dims
            [1, 2],
            # expected
            "const_1_2",
        ),
        (
            # dims
            [2, 1],
            # expected
            "const_1_2",
        ),
    ],
)
def test_const_col(dims: List[int], expected: str) -> None:
    actual = const_col(dims)
    assert actual == expected


@pytest.mark.parametrize(
    "d, expected",
    [
        (
            # d
            1,
            # expected
            "dim_1",
        ),
        (
            # d
            2,
            # expected
            "dim_2",
        ),
    ],
)
def test_dim_col(d: int, expected: str) -> None:
    actual = dim_col(d)
    assert actual == expected


SIZE = 4
N_DIMS = 3


class TestGenerateTestDataFrame:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return generate_test_dataframe(n_dims=N_DIMS, size=SIZE)

    @pytest.fixture(params=list(range(N_DIMS)))
    def d(self, request: SubRequest) -> int:
        i = request.param
        assert isinstance(i, int)
        return i

    @pytest.fixture(
        params=list(
            itertools.chain(
                *(
                    itertools.combinations(range(N_DIMS), r)
                    for r in range(1, N_DIMS + 1)
                )
            )
        )
    )
    def dims(self, request: SubRequest) -> List[int]:
        return cast(List[int], request.param)

    def test_len(self, df: pd.DataFrame) -> None:
        assert len(df) == SIZE ** N_DIMS

    def test_index(self, df: pd.DataFrame) -> None:
        pdt.assert_index_equal(df.index, pd.RangeIndex(0, len(df)))
        assert isinstance(df.index, pd.RangeIndex)

    def test_dim_nunique(self, df: pd.DataFrame, d: int) -> None:
        assert df[dim_col(d)].nunique() == SIZE

    def test_dim_value_counts(self, df: pd.DataFrame, d: int) -> None:
        assert (df[dim_col(d)].value_counts() == SIZE ** (N_DIMS - 1)).all()

    def test_dims_sorted(self, df: pd.DataFrame, d: int) -> None:
        delta = df[dim_col(d)].values[1:] - df[dim_col(d)].values[:-1]
        assert ((delta == 0) | (delta == 1) | (delta == -(SIZE - 1))).all()

    def test_const_nunique(self, df: pd.DataFrame, dims: List[int]) -> None:
        assert df[const_col(dims)].nunique() == SIZE ** len(dims)

    def test_const_value_counts(self, df: pd.DataFrame, dims: List[int]) -> None:
        assert (
            df[const_col(dims)].value_counts() == SIZE ** (N_DIMS - len(dims))
        ).all()

    def test_cols_sorted(self, df: pd.DataFrame) -> None:
        assert list(df.columns) == sorted(df.columns)


def test_generate_example() -> None:
    df = generate_example()
    assert len(df) == 2000 ** 2
    assert list(df.columns) == [
        "date",
        "month",
        "year",
        "city",
        "country",
        "avg_temp",
        "rain",
        "mood",
    ]
