"""
Functionality useful for testing and documentation.
"""
import itertools
from typing import Iterable

import numpy as np
import pandas as pd


def dim_col(d: int) -> str:
    """
    Name of an dimension columns.

    Parameters
    ----------
    d
        Dimension number.

    Returns
    -------
    name: str
        Dimension name.

    Example
    -------
    >>> from rle_array.testing import dim_col
    >>> dim_col(1)
    'dim_1'
    """
    return f"dim_{d}"


def const_col(dims: Iterable[int]) -> str:
    """
    Name of an constant columns.

    Parameters
    ----------
    dims
        Dimensions, that describe the column content.

    Returns
    -------
    name: str
        Column name.

    Example
    -------
    >>> from rle_array.testing import const_col
    >>> const_col([1, 2])
    'const_1_2'
    >>> const_col([2, 1])
    'const_1_2'
    """
    dims = sorted(dims)
    dims_str = [str(d) for d in dims]
    return f"const_{'_'.join(dims_str)}"


def _insert_sorted(df: pd.DataFrame, column: str, value: np.ndarray) -> None:
    pos = 0
    for i, c in enumerate(df.columns):
        if c > column:
            break
        pos = i + 1
    df.insert(pos, column, value)


def _setup_dim_df(n_dims: int, size: int) -> pd.DataFrame:
    elements = np.arange(size ** n_dims)
    df = pd.DataFrame(index=elements)
    for i in range(n_dims):
        _insert_sorted(df, dim_col(i), (elements // (size ** i)) % size)
    return df


def _add_const_cols(df: pd.DataFrame, n_dims: int, size: int) -> pd.DataFrame:
    for dims in itertools.chain(
        *(itertools.combinations(range(n_dims), l + 1) for l in range(n_dims))
    ):
        data = None
        for d in dims:
            if data is None:
                data = df[dim_col(d)].values
            else:
                data = data * size + df[dim_col(d)].values
        _insert_sorted(df, const_col(dims), data)
    return df


def generate_test_dataframe(n_dims: int, size: int) -> pd.DataFrame:
    """
    Generates testing data.

    Parameters
    ---------
    n_dims
        Number of dimensions of test cube.
    size
        Size of every dimension (edge length).

    Returns
    -------
    df: pd.DataFrame
        Test DataFrame.
    """
    df = _setup_dim_df(n_dims, size)
    df = _add_const_cols(df, n_dims, size)
    return df


def generate_example() -> pd.DataFrame:
    """
    Generate example DataFrame for documentation purposes.

    Returns
    -------
    df: pd.DataFrame
        Example DataFrame.
    """
    rng = np.random.RandomState(1234)

    df = generate_test_dataframe(n_dims=2, size=2000)
    df["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(df["dim_0"], unit="D")
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["city"] = "city_" + df["dim_1"].astype("str")
    df["country"] = "country_" + (df["dim_1"] // 500).astype("str")
    df["avg_temp"] = (
        rng.normal(loc=10.0, scale=5.0, size=len(df))
        .round(decimals=1)
        .astype(np.float32)
    )
    df["rain"] = rng.rand(len(df)) > 0.9
    df["mood"] = "ok"
    df.loc[(~df["rain"]) & (df["avg_temp"] > 15), "mood"] = "great"
    df.loc[(df["rain"]) & (df["avg_temp"] < 5), "mood"] = "sad"
    return df[["date", "month", "year", "city", "country", "avg_temp", "rain", "mood"]]
