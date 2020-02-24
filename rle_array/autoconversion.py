import warnings
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype

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


def auto_convert_to_rle(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Auto-convert given DataFrame to RLE compressed DataFrame.

    .. important::

        Datetime columns are currently not compressed due to pandas not supporting them.

    Parameters
    ----------
    df
        Input DataFrame, may already contain RLE columns.
    threshold
        Compression threshold (``1`` compresses all, ``0.5`` compresses if at least 50% memory are safed, ``0.0`` does
        not compress at all).
    """
    if (threshold < 0.0) or (threshold > 1.0):
        raise ValueError(f"threshold ({threshold}) must be in [0, 1]")

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
                    f"Column {col} is would use a DatetimeBlock and can currently not be RLE compressed."
                )
            else:
                array_rle = RLEArray._from_sequence(
                    scalars=array_orig, dtype=dtype, copy=True
                )
                if threshold == 1.0:
                    array_target = array_rle
                elif threshold > 0:
                    if (len(array_orig) == 0) or (
                        array_rle.nbytes / array_orig.nbytes <= threshold
                    ):
                        array_target = array_rle

        data[col] = array_target

    return pd.DataFrame(data, index=index)
