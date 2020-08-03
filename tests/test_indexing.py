import pytest

from rle_array import RLEArray


def test_fail_two_dim_indexing() -> None:
    array = RLEArray._from_sequence(range(10))
    with pytest.raises(
        NotImplementedError,
        match="__getitem__ does currently only work w/ a single parameter",
    ):
        array[1, 2]
