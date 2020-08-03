import numpy as np
import pytest

from rle_array import RLEArray
from rle_array.types import POSITIONS_DTYPE


def test_valid() -> None:
    RLEArray(
        data=np.asarray([1.0, 2.0]),
        positions=np.asarray([10, 20], dtype=POSITIONS_DTYPE),
    )


def test_data_invalid_type() -> None:
    with pytest.raises(TypeError, match="data must be an ndarray but is int"):
        RLEArray(data=1, positions=np.asarray([10, 20], dtype=POSITIONS_DTYPE))


def test_positions_invalid_type() -> None:
    with pytest.raises(TypeError, match="positions must be an ndarray but is int"):
        RLEArray(data=np.asarray([1.0, 2.0]), positions=1)


def test_data_invalid_dims() -> None:
    with pytest.raises(
        ValueError, match="data must be an 1-dimensional ndarray but has 2 dimensions"
    ):
        RLEArray(
            data=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            positions=np.asarray([10, 20], dtype=POSITIONS_DTYPE),
        )


def test_positions_invalid_dims() -> None:
    with pytest.raises(
        ValueError,
        match="positions must be an 1-dimensional ndarray but has 2 dimensions",
    ):
        RLEArray(
            data=np.asarray([1.0, 2.0]),
            positions=np.asarray([[10, 20], [30, 40]], dtype=POSITIONS_DTYPE),
        )


def test_positions_invalid_dtype() -> None:
    with pytest.raises(
        ValueError, match="positions must have dtype int64 but has uint64"
    ):
        RLEArray(
            data=np.asarray([1.0, 2.0]), positions=np.asarray([10, 20], dtype=np.uint64)
        )


def test_different_lengths() -> None:
    with pytest.raises(
        ValueError, match="data and positions must have same length but have 3 and 2"
    ):
        RLEArray(
            data=np.asarray([1.0, 2.0, 3.0]),
            positions=np.asarray([10, 20], dtype=POSITIONS_DTYPE),
        )


def test_not_sorted_1() -> None:
    with pytest.raises(ValueError, match="positions must be strictly sorted"):
        RLEArray(
            data=np.asarray([1.0, 2.0]),
            positions=np.asarray([10, 9], dtype=POSITIONS_DTYPE),
        )


def test_not_sorted_2() -> None:
    with pytest.raises(ValueError, match="positions must be strictly sorted"):
        RLEArray(
            data=np.asarray([1.0, 2.0]),
            positions=np.asarray([10, 10], dtype=POSITIONS_DTYPE),
        )
