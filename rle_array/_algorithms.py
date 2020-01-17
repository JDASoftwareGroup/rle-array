from typing import Any, Iterator, List, Optional, Tuple

import numba
import numpy as np
import pandas as pd

from ._slicing import NormalizedSlice
from .types import POSITIONS_DTYPE


def calc_lengths(positions: np.ndarray) -> np.ndarray:
    """
    Calculate lengths of runs.

    Parameters
    ----------
    positions:
        End positions of runs.

    Returns
    -------
    lengths:
        Lengths of runs.
    """
    return np.concatenate([positions[:1], positions[1:] - positions[:-1]])


def compress(scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress given array of scalars to RLE.

    Parameters
    ----------
    scalars:
        Scalars to compress.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    if len(scalars) == 0:
        return (scalars, np.array([], dtype=POSITIONS_DTYPE))

    changes = detect_changes(scalars)

    data = np.concatenate([scalars[:-1][changes], scalars[-1:]])
    positions = np.concatenate(
        [np.where(changes)[0] + 1, np.asarray([len(scalars)], dtype=POSITIONS_DTYPE)]
    )
    return (data, positions)


def concat(
    data_parts: List[np.ndarray], positions_parts: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate RLE data.

    Parameters
    ----------
    data_parts:
        For each part: Data at start of reach run.
    positions_parts:
        For each part: End positions of runs.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    assert len(data_parts) == len(positions_parts)
    if len(data_parts) == 0:
        return (np.array([]), np.array([], dtype=POSITIONS_DTYPE))

    lengths = np.asarray([get_len(positions) for positions in positions_parts])
    offsets = np.roll(np.cumsum(lengths), 1)
    offsets[0] = 0

    data = np.concatenate([data for data in data_parts])
    positions = np.concatenate(
        [positions + o for positions, o in zip(positions_parts, offsets)]
    )

    data, positions = recompress(data, positions)
    return (data, positions)


def decompress(
    data: np.ndarray, positions: np.ndarray, dtype: Optional[Any] = None
) -> np.ndarray:
    """
    Decompress RLE data.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    dtype:
        Optional dtype for conversion.

    Returns
    -------
    scalars:
        Scalars, decompressed.
    """
    lengths = calc_lengths(positions)
    return np.repeat(data.astype(dtype), lengths)


def detect_changes(scalars: np.ndarray) -> np.ndarray:
    """
    Detect changes in array of scalars. These changes can be used as boundaries for RLE-runs.

    Parameters
    ----------
    scalars:
        Scalars to compress.

    Returns
    -------
    changes:
        Change points (boolean mask).
    """
    nulls = pd.isna(scalars)
    identical = (scalars[1:] == scalars[:-1]) | (nulls[1:] & nulls[:-1])
    return ~identical


def dropna(data: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop NULL-values from RLE data.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    mask = pd.notnull(data)
    data = data[mask]
    lenghts = calc_lengths(positions)
    positions = (
        positions
        - np.cumsum(lenghts * (~mask).astype(POSITIONS_DTYPE), dtype=POSITIONS_DTYPE)
    )[mask]
    return (data, positions)


def find_single_index(data: np.ndarray, positions: np.ndarray, i: int) -> Any:
    """
    Find single element in RLE data.

    .. important:
        This function does NOT handle negative indices.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.

    Returns
    -------
    element:
        Found element.

    Raises
    ------
    IndexError: In case of an out-of-bounds index request.
    """
    if (i < 0) or (i > get_len(positions)):
        raise IndexError(f"{i} out of bounds")
    return data[np.searchsorted(positions, i, side="right")]


def find_slice(
    data: np.ndarray, positions: np.ndarray, s: slice
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get slice of RLE data.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    length = get_len(positions)
    s_norm = NormalizedSlice.from_slice(length, s)

    start, stop, step = s_norm.start, s_norm.stop, s_norm.step
    invert = False
    if s_norm.step < 0:
        invert = True
        start, stop = stop + 1, start + 1
        step = abs(step)

    if start == 0:
        idx_start = 0
    elif start >= length:
        idx_start = len(positions)
    else:
        idx_start = np.searchsorted(positions, start, side="right")
    if stop == 0:
        idx_stop = 0
    elif stop >= length:
        idx_stop = len(positions)
    else:
        idx_stop = np.searchsorted(positions, stop, side="left") + 1

    data = data[idx_start:idx_stop]
    positions = positions[idx_start:idx_stop] - start
    if len(positions) > 0:
        positions[-1] = stop - start

    if invert:
        lenghts = calc_lengths(positions)
        lenghts = lenghts[::-1]
        positions = np.cumsum(lenghts)
        data = data[::-1]

    if step != 1:
        positions = ((positions - 1) // step) + 1

        mask = np.empty(len(positions), dtype=bool)
        if len(positions) > 0:
            mask[0] = True
        mask[1:] = positions[1:] != positions[:-1]

        data = data[mask]
        positions = positions[mask]

        data, positions = recompress(data, positions)

    return (data, positions)


def gen_iterator(data: np.ndarray, positions: np.ndarray) -> Iterator[Any]:
    """
    Generate iterator over RLE data.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.

    Returns
    -------
    it:
        Iterator over uncompressed values.
    """
    old_p = 0
    for x, p in zip(data, positions):
        for _ in range(p - old_p):
            yield x
        old_p = p


def get_len(positions: np.ndarray) -> int:
    """
    Get length of RLE data.

    Parameters
    ----------
    positions:
        End positions of runs.

    Returns
    -------
    len:
        Length.
    """
    if len(positions) > 0:
        return int(positions[-1])
    else:
        return 0


def recompress(
    data: np.ndarray, positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try to compress RLE data even more.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    changes = detect_changes(data)

    data = np.concatenate([data[:-1][changes], data[-1:]])
    positions = np.concatenate([positions[:-1][changes], positions[-1:]])
    return (data, positions)


@numba.jit((numba.int64[:], numba.int64[:]), nopython=True, cache=True, nogil=True)
def _take_kernel(
    positions: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(indices)

    # pre-allocate output buffers
    result_data_idx = np.empty(n, dtype=POSITIONS_DTYPE)
    result_positions = np.empty(n, dtype=POSITIONS_DTYPE)

    current = -2
    run_start = 0
    run_stop = 0
    out_count = 0
    for pos in range(n):
        i = indices[pos]
        if i == -1:
            # fill
            idx = -1
        elif current >= 0 and (run_start <= i) and (i < run_stop):
            # great, same RLE-run
            idx = current
        else:
            # run full search
            idx = np.searchsorted(positions, i, side="right")

        # flush?
        if idx != current:
            if current != -2:
                result_data_idx[out_count] = current
                result_positions[out_count] = pos
                out_count += 1
            current = idx

            if current > 0:
                run_start = positions[current - 1]
            else:
                run_start = 0

            if current >= 0:
                run_stop = positions[current]

    # flush?
    if current != -2:
        result_data_idx[out_count] = current
        result_positions[out_count] = n
        out_count += 1

    # return clean-cut outputs
    return result_data_idx[:out_count].copy(), result_positions[:out_count].copy()


def take(
    data: np.ndarray,
    positions: np.ndarray,
    indices: np.ndarray,
    allow_fill: bool,
    fill_value: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take values from RLE array.

    Parameters
    ----------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    indices:
        Indices to take. If ``allow_fill`` is set, the only negative element allowed is ``-1``. If ``allow_fill`` is not
        set, then negative entries will be counted from the end of the array.
    allow_fill:
        If filling with missing values is allowed. In that case, ``-1`` in ``indices`` will be filled with
        ``fill_value``.
    fill_value:
        Fill-value in case ``allow_fill`` is set.

    Returns
    -------
    data:
        Data at start of reach run.
    positions:
        End positions of runs.
    """
    length = get_len(positions)
    indices = indices.copy()

    if (length == 0) and ((np.any(indices != -1) and allow_fill) or not allow_fill):
        raise IndexError("cannot do a non-empty take")

    if allow_fill:
        out_of_bounds_mask = indices < -1
        if np.any(out_of_bounds_mask):
            raise ValueError(f"{indices[out_of_bounds_mask][0]}")
        min_idx_allowed = -1
    else:
        indices[indices < 0] += length
        min_idx_allowed = 0

    out_of_bounds_mask = (indices < min_idx_allowed) | (indices >= length)
    if np.any(out_of_bounds_mask):
        raise IndexError(f"{indices[out_of_bounds_mask][0]} out of bounds")

    result_data_idx, result_positions = _take_kernel(positions, indices)

    result_data_mask = result_data_idx != -1
    result_data = np.empty(len(result_data_idx), dtype=data.dtype)
    result_data[result_data_mask] = data[result_data_idx[result_data_mask]]
    if np.any(~result_data_mask):
        result_data[~result_data_mask] = fill_value

    return recompress(result_data, result_positions)


@numba.jit((numba.int64[:], numba.int64[:]), nopython=True, cache=True, nogil=True)
def _extend_positions_kernel(
    positions1: np.ndarray, positions2: np.ndarray
) -> np.ndarray:
    n1 = len(positions1)
    n2 = len(positions2)

    # pre-allocate output buffers
    result = np.empty(n1 + n2, dtype=POSITIONS_DTYPE)

    i_out = 0
    i1 = 0
    i2 = 0

    while (i1 < n1) and (i2 < n2):
        x1 = positions1[i1]
        x2 = positions2[i2]

        if x1 == x2:
            result[i_out] = x1
            i1 += 1
            i2 += 1
        elif x1 < x2:
            result[i_out] = x1
            i1 += 1
        else:
            # x2 < x1
            result[i_out] = x2
            i2 += 1

        i_out += 1

    while i1 < n1:
        result[i_out] = positions1[i1]
        i1 += 1
        i_out += 1

    while i2 < n2:
        result[i_out] = positions2[i2]
        i2 += 1
        i_out += 1

    # return clean-cut output
    return result[:i_out].copy()


def extend_positions(positions1: np.ndarray, positions2: np.ndarray) -> np.ndarray:
    """
    Create union of two position arrays.

    Parameters
    ----------
    positions1
        First position array.
    positions2
        Second position array.

    Returns
    -------
    extended_positions
        Sorted position array that contains all entries from input arrays (without duplicates).
    """
    return _extend_positions_kernel(positions1, positions2)


@numba.jit(nopython=True, cache=True, nogil=True)
def _extend_data_kernel(
    data: np.ndarray, positions: np.ndarray, extended_positions: np.ndarray
) -> np.ndarray:
    n = extended_positions.shape[0]
    extended_array = np.empty(n, dtype=data.dtype)

    k = 0  # current index for data/positions
    for i in range(n):
        if extended_positions[i] > positions[k]:
            k += 1
        extended_array[i] = data[k]

    return extended_array


def extend_data(
    data: np.ndarray, positions: np.ndarray, extended_positions: np.ndarray
) -> np.ndarray:
    """
    Extend data array to match new positions.

    Parameters
    ----------
    data
        Data at start of reach run.
    positions
        End positions of runs.
    extended_positions
        Extended position array (superset of ``positions``). See :func:`extend_positions`.

    Returns
    -------
    extended_data
        Extended data array.
    """
    return _extend_data_kernel(data, positions, extended_positions)
