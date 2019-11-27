import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from rle_array import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.fixture
def series_orig():
    return pd.Series([1, 1, 2, 3, 3], dtype=int)


@pytest.fixture
def array_orig(series_orig):
    return series_orig.values


@pytest.fixture
def series_rle(series_orig):
    return series_orig.astype(RLEDtype(series_orig.dtype))


@pytest.fixture
def array_rle(series_rle):
    return series_rle.values


@pytest.fixture(params=["series", "array"])
def mode(request):
    return request.param


@pytest.fixture
def object_orig(series_orig, array_orig, mode):
    if mode == "series":
        return series_orig
    elif mode == "array":
        return array_orig
    else:
        raise ValueError(f"Unknown mode {mode}")


@pytest.fixture
def object_rle(series_rle, array_rle, mode):
    if mode == "series":
        return series_rle
    elif mode == "array":
        return array_rle
    else:
        raise ValueError(f"Unknown mode {mode}")


@pytest.fixture
def comp(mode):
    if mode == "series":
        return pdt.assert_series_equal
    elif mode == "array":
        return npt.assert_array_equal
    else:
        raise ValueError(f"Unknown mode {mode}")


def test_sum(object_orig, object_rle, comp):
    elements_orig = [object_orig, object_orig]
    elements_rle = [object_rle, object_rle]
    elements_mixed = [object_rle, object_orig]

    result_orig = sum(elements_orig)
    result_rle = sum(elements_rle)
    result_mixed = sum(elements_mixed)

    result_converted1 = result_rle.astype(int)
    comp(result_orig, result_converted1)

    result_converted2 = result_mixed.astype(int)
    comp(result_orig, result_converted2)
