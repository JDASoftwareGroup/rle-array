import numpy as np
import numpy.testing as npt
import pytest

from rle_array import RLEArray

pytestmark = pytest.mark.filterwarnings("ignore:performance")


def test_view_raises_differnt_dtype():
    orig = RLEArray._from_sequence(np.arange(10))
    with pytest.raises(ValueError, match="Cannot create view with different dtype"):
        orig.view(np.int8)


@pytest.mark.parametrize("use_numpy_dtype", [False, True])
def test_plain_view(use_numpy_dtype):
    orig = RLEArray._from_sequence(np.arange(10))

    dtype = orig.dtype
    if use_numpy_dtype:
        dtype = dtype._dtype
    view = orig.view(dtype)

    assert view is not orig
    npt.assert_array_equal(orig, view)

    orig[[0, 1]] = [100, 101]
    view[[0, 8, 9]] = [1000, 108, 109]

    result = RLEArray._from_sequence([1000, 101, 2, 3, 4, 5, 6, 7, 108, 109])

    npt.assert_array_equal(orig, result)
    npt.assert_array_equal(orig, view)


def test_view_tree():
    # o-->1-+->11
    #       +->12
    orig = RLEArray._from_sequence(np.arange(10))

    view1 = orig.view(orig.dtype)
    view11 = view1.view(view1.dtype)
    view12 = view1.view(view1.dtype)

    assert view1 is not orig
    assert view11 is not orig
    assert view12 is not orig
    assert view11 is not view1
    assert view12 is not view1
    assert view11 is not view12
    npt.assert_array_equal(orig, view1)
    npt.assert_array_equal(orig, view11)
    npt.assert_array_equal(orig, view12)

    view11[[8, 9]] = [108, 109]
    view1[[0, 1, 9]] = [100, 101, 1009]

    result = RLEArray._from_sequence([100, 101, 2, 3, 4, 5, 6, 7, 108, 1009])

    npt.assert_array_equal(orig, result)
    npt.assert_array_equal(orig, view1)
    npt.assert_array_equal(orig, view11)
    npt.assert_array_equal(orig, view12)


def test_slicing():
    N = 100
    orig_np = np.arange(N)
    orig_rle = RLEArray._from_sequence(orig_np)

    ops = [
        slice(None, None, None),
        slice(1, -3, 2),
        slice(None, None, -1),
        slice(None, None, -1),
        slice(3, 4, -3),
    ]

    arrays_np = [orig_np]
    arrays_rle = [orig_rle]
    for i, o in enumerate(ops):
        last_np = arrays_np[-1]
        last_rle = arrays_rle[-1]
        npt.assert_array_equal(last_np, last_rle)

        sub_np = last_np[o]
        sub_rle = last_rle[o]

        assert sub_np is not last_np
        assert sub_rle is not last_rle
        npt.assert_array_equal(sub_np, sub_rle)

        delta = np.arange(len(sub_np)) * (N ** i)

        # `+=` seems to convert sub_rle from RLEArray to ndarray?
        sub_np[:] = sub_np + delta
        sub_rle[:] = sub_rle + delta

        arrays_np.append(sub_np)
        arrays_rle.append(sub_rle)

    for arr_np, arr_rle in zip(arrays_np, arrays_rle):
        npt.assert_array_equal(arr_np, arr_rle)
