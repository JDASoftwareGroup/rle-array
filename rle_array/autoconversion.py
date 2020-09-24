import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.errors import PerformanceWarning

from .array import RLEArray
from .dtype import RLEDtype


def _is_rle_dtype(dtype: Union[np.dtype, ExtensionDtype]) -> bool:
    """
    Checks if the given dtype is already RLE compressed.

    Parameters
    ----------
    dtype
        Input dtype.
    """
    return isinstance(dtype, RLEDtype)


def _uses_datetimeblock(dtype: Union[np.dtype, ExtensionDtype]) -> bool:
    """
    Detects if the RLEArray would use a pandas ``DatetimeBlock``.

    It seems to be a bug in pandas that it cannot deal with datetime extension arrays.

    Parameters
    ----------
    dtype
        Dtype of the original, uncompressed array.
    """
    vtype = dtype.type
    return issubclass(vtype, np.datetime64)


def auto_convert_to_rle(
    df: pd.DataFrame, threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Auto-convert given DataFrame to RLE compressed DataFrame.

    .. important::

        Datetime columns are currently not compressed due to pandas not supporting them.

    Please note that RLE can, under some circumstances, require MORE memory than the uncompressed data. It is not
    advisable to set ``threshold`` to a value larger than 1 except for testing purposes.

    Parameters
    ----------
    df
        Input DataFrame, may already contain RLE columns. This input data MIGHT not be copied!
    threshold
        Compression threshold, e.g.:

        - ``None``: compress all
        - ``1.0`` compresses only if RLE does NOT take up more space
        - ``0.5`` compresses if at least 50% memory are safed
        - ``0.0`` do not compress at all

    Raises
    ------
    ValueError
        If threshold is negative.
    """
    if (threshold is not None) and (threshold < 0.0):
        raise ValueError(f"threshold ({threshold}) must be non-negative")

    index = df.index

    data = {}
    for col in df.columns:
        series = df[col]
        array_orig = series.array

        array_target = array_orig

        dtype = series.dtype

        if not _is_rle_dtype(dtype):
            if _uses_datetimeblock(dtype):
                warnings.warn(
                    f"Column {col} would use a DatetimeBlock and can currently not be RLE compressed."
                )
            else:
                array_rle = RLEArray._from_sequence(
                    scalars=array_orig, dtype=dtype, copy=True
                )
                if threshold is None:
                    array_target = array_rle
                elif threshold > 0:
                    if (len(array_orig) == 0) or (
                        array_rle.nbytes / array_orig.nbytes <= threshold
                    ):
                        array_target = array_rle

        data[col] = array_target

    return pd.DataFrame(data, index=index)


def decompress(df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Decompress all RLE columns in the provided DataFrame.

    Parameters
    ----------
    df
        Input DataFrame. This input data MIGHT not be copied!
    """
    index = df.index

    data = {}
    for col in df.columns:
        series = df[col]
        array = series.array
        dtype = series.dtype

        if _is_rle_dtype(dtype):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=PerformanceWarning)
                array = array.astype(dtype._dtype)

        data[col] = array

    return pd.DataFrame(data, index=index)
