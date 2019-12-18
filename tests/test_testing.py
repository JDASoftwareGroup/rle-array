import itertools

import pytest

from rle_array.testing import (
    const_col,
    dim_col,
    generate_example,
    generate_test_dataframe,
)


@pytest.mark.parametrize(
    "dims,expected",
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
def test_const_col(dims, expected):
    actual = const_col(dims)
    assert actual == expected


@pytest.mark.parametrize(
    "d,expected",
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
def test_dim_col(d, expected):
    actual = dim_col(d)
    assert actual == expected


SIZE = 2
N_DIMS = 3


class TestGenerateTestDataFrame:
    @pytest.fixture
    def df(self):
        return generate_test_dataframe(n_dims=N_DIMS, size=SIZE)

    @pytest.fixture(params=list(range(N_DIMS)))
    def d(self, request):
        return request.param

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
    def dims(self, request):
        yield request.param

    def test_len(self, df):
        assert len(df) == SIZE ** N_DIMS

    def test_dim_nunique(self, df, d):
        assert df[dim_col(d)].nunique() == SIZE

    def test_const_nunique(self, df, dims):
        assert df[const_col(dims)].nunique() == SIZE ** len(dims)

    def test_cols_sorted(self, df):
        assert list(df.columns) == sorted(df.columns)


def test_generate_example():
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
