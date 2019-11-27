import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension import base

from rle_array import RLEArray, RLEDtype
from rle_array.types import POSITIONS_DTYPE

pytestmark = pytest.mark.filterwarnings("ignore:performance")


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names
    """
    return request.param


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture
def data():
    """Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return RLEArray(
        data=np.asarray([13, -1, -2, 42], dtype=np.float32),
        positions=np.asarray([1, 2, 4, 100], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    return RLEArray(
        data=np.asarray([2.0, np.nan, 1.0, 2.0, 3.0], dtype=np.float32),
        positions=np.asarray([2, 4, 6, 7, 8], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    return RLEArray(
        data=np.asarray([2.0, 3.0, 1.0], dtype=np.float32),
        positions=np.asarray([1, 2, 3], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    return RLEArray(
        data=np.asarray([2.0], dtype=np.float32),
        positions=np.asarray([100], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return RLEArray(
        data=np.asarray([np.nan, 42], dtype=np.float32),
        positions=np.asarray([1, 2], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return RLEArray(
        data=np.asarray([2.0, np.nan, 1.0], dtype=np.float32),
        positions=np.asarray([1, 2, 3], dtype=POSITIONS_DTYPE),
    )


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return RLEDtype(np.float32)


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    return lambda x, y: np.isnan(x) and np.isnan(y)


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return np.nan


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    frame_scalar_exc = None
    series_array_exc = None
    series_scalar_exc = None

    def test_error(self):
        pytest.skip("upstream test is broken?")

    def _check_op(self, s, op, other, op_name, exc=NotImplementedError):
        # upstream version checks dtype -> we return an RLEDtype
        if exc is None:
            result = op(s, other)
            expected = s.combine(other, op)
            self.assert_series_equal(result, expected, check_dtype=False)
        else:
            with pytest.raises(exc):
                op(s, other)


class TestBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestCasting(base.BaseCastingTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestDtype(base.BaseDtypeTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    def test_combine_le(self):
        pytest.skip("upstream test is broken?")


class TestMissing(base.BaseMissingTests):
    def test_isna(self):
        pytest.skip("upstream test is broken")


class TestNumericReduce(base.BaseNumericReduceTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    def test_concat_mixed_dtypes(self):
        pytest.skip("upstream test is broken?")


class TestSetitem(base.BaseSetitemTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        # upstream version looks pretty broken...
        op = self.get_op_from_name(op_name)
        if op_name == "__eq__":
            assert getattr(data, op_name)(other) is NotImplemented
            assert not op(s, other).all()
        else:
            assert getattr(data, op_name)(other) is NotImplemented

    def test_compare_scalar(self, data, all_compare_operators):
        pytest.skip("upstream test is broken: comparison with scalar works")
