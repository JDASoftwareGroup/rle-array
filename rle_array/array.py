import logging
import operator
import warnings
from collections import namedtuple
from copy import copy
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Union
from weakref import WeakSet, ref

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.arrays import BooleanArray, IntegerArray, StringArray
from pandas.core import ops
from pandas.core.dtypes.common import is_array_like
from pandas.core.dtypes.generic import ABCIndexClass, ABCSeries
from pandas.core.dtypes.inference import is_scalar
from pandas.core.dtypes.missing import isna
from pandas.errors import PerformanceWarning

from ._algorithms import (
    calc_lengths,
    compress,
    concat,
    decompress,
    dropna,
    extend_data,
    extend_positions,
    find_single_index,
    find_slice,
    gen_iterator,
    get_len,
    recompress,
    take,
)
from ._operators import rev
from ._slicing import NormalizedSlice
from .dtype import RLEDtype
from .types import POSITIONS_DTYPE

_logger = logging.getLogger(__name__)


def _normalize_arraylike_indexing(arr: Any, length: int) -> np.ndarray:
    """
    Normalize array-like index arguments for ``__getitem__`` and ``__setitem__``.

    This is required since pandas can pass us many different types with potentially nullable data.

    Parameters
    ----------
    arr
        Index argument passed to ``__getitem__`` and ``__setitem__`` if arraylike.
    length
        Array length.
    """
    if isinstance(arr, BooleanArray):
        result = np.asarray(arr.fillna(False), dtype=bool)
    elif isinstance(arr, IntegerArray):
        try:
            return np.asarray(arr, dtype=int)
        except ValueError:
            raise ValueError(
                "Cannot index with an integer indexer containing NA values"
            )
    elif isinstance(arr, RLEArray):
        result = np.asarray(arr, dtype=arr.dtype._dtype)
    elif isinstance(arr, list):
        if any((pd.isna(x) for x in arr)):
            raise ValueError(
                "Cannot index with an integer indexer containing NA values"
            )
        result = np.asarray(arr)
    else:
        result = np.asarray(arr)

    if (result.dtype == np.bool_) and (len(result) != length):
        raise IndexError("Indexer has wrong length")

    return result


class _ViewAnchor:
    """
    Anchor object that references a RLEArray because it is not hashable.
    """

    def __init__(self, array: "RLEArray") -> None:
        self.array = ref(array)

    def __hash__(self) -> int:
        return id(self.array)


class _ViewMaster:
    """
    Collection of all views to an array.

    This tracks the original data as well as all views.
    """

    def __init__(self, data: np.ndarray, positions: np.ndarray):
        self.data = data
        self.positions = positions
        self.views: WeakSet[_ViewAnchor] = WeakSet()

    @classmethod
    def register_first(cls, array: "RLEArray") -> "_Projection":
        """
        Register array with new master.

        The array must not have a view master yet!
        """
        assert getattr(array, "_projection", None) is None

        projection = _Projection(
            projection_slice=None,
            master=cls(data=array._data, positions=array._positions),
        )
        projection.master.views.add(array._view_anchor)
        return projection

    def register_change(
        self, array: "RLEArray", projection_slice: Optional[slice]
    ) -> None:
        """
        Re-register array with new view-master.

        The array must only be registered with a single, no orphan master!
        """
        # ensure the array is only registered with another orphan master
        assert array._projection is not None
        assert array._projection.projection_slice is None
        assert array._projection.master is not self
        assert len(array._projection.master.views) == 1
        assert array._view_anchor not in self.views

        array._projection = _Projection(projection_slice=projection_slice, master=self)
        self.views.add(array._view_anchor)

    def modify(self, data: np.ndarray, positions: np.ndarray) -> None:
        """
        Modify the original (unprojected) data and populate change to all views.
        """
        self.data = data
        self.positions = positions

        for view in self.views:
            array = view.array()
            assert array is not None
            assert array._projection is not None
            assert array._projection.master is self

            if array._projection.projection_slice is not None:
                data2, positions2 = find_slice(
                    data=self.data,
                    positions=self.positions,
                    s=array._projection.projection_slice,
                )
            else:
                data2, positions2 = self.data, self.positions

            array._data = data2
            array._positions = positions2


_Projection = namedtuple("_Projection", ["master", "projection_slice"])


class RLEArray(ExtensionArray):
    """
    Run-length encoded array.

    Parameters
    ----------
    data
        Data for each run. Must be a one-dimensional. All Pandas-supported dtypes are supported.
    positions
        End-positions for each run. Must be one-dimensional and must have same length as ``data``. dtype must be
        ``POSITIONS_DTYPE``.
    """

    _HANDLED_TYPES = tuple(
        t for types in np.sctypes.values() for t in types if t is not object
    ) + (np.ndarray, list, tuple, int, float, complex)

    # For comparisons, so that numpy uses our implementation.
    __array_priority__ = 1000

    def __init__(self, data: np.ndarray, positions: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be an ndarray but is {type(data).__name__}")
        if not isinstance(positions, np.ndarray):
            raise TypeError(
                f"positions must be an ndarray but is {type(positions).__name__}"
            )
        if data.ndim != 1:
            raise ValueError(
                f"data must be an 1-dimensional ndarray but has {data.ndim} dimensions"
            )
        if positions.ndim != 1:
            raise ValueError(
                f"positions must be an 1-dimensional ndarray but has {positions.ndim} dimensions"
            )
        if positions.dtype != POSITIONS_DTYPE:
            raise ValueError(
                f"positions must have dtype {POSITIONS_DTYPE.__name__} but has {positions.dtype}"
            )
        if len(data) != len(positions):
            raise ValueError(
                f"data and positions must have same length but have {len(data)} and {len(positions)}"
            )
        if np.any(positions[1:] <= positions[:-1]):
            raise ValueError("positions must be strictly sorted")

        _logger.debug(
            "RLEArray.__init__(data=%s(len=%r, dtype=%r), positions=%s(len=%r, dtype=%r))",
            type(data).__name__,
            len(data),
            data.dtype,
            type(positions).__name__,
            len(positions),
            positions.dtype,
        )

        self._dtype = RLEDtype(data.dtype)
        self._data = data
        self._positions = positions
        self._setup_view_system()

    def _setup_view_system(self) -> None:
        """
        Setup any view-related tracking parts.

        Must be called after initialization or unpickling.
        """
        self._view_anchor = _ViewAnchor(self)
        self._projection = _ViewMaster.register_first(self)

    def __getstate__(self) -> Dict[str, Any]:
        state = copy(self.__dict__)
        del state["_view_anchor"]
        del state["_projection"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._setup_view_system()

    @property
    def _lengths(self) -> Any:
        return calc_lengths(self._positions)

    @classmethod
    def _from_sequence(
        cls, scalars: Any, dtype: Any = None, copy: bool = False
    ) -> "RLEArray":
        _logger.debug(
            "RLEArray._from_sequence(scalars=%s(...), dtype=%r, copy=%r)",
            type(scalars).__name__,
            dtype,
            copy,
        )
        if isinstance(dtype, RLEDtype):
            dtype = dtype._dtype

        scalars = np.asarray(scalars, dtype=dtype)
        data, positions = compress(scalars)
        return RLEArray(data=data, positions=positions)

    @classmethod
    def _from_factorized(cls, data: Any, original: "RLEArray") -> "RLEArray":
        _logger.debug("RLEArray._from_factorized(...)")
        return cls._from_sequence(np.asarray(data, dtype=original.dtype._dtype))

    def __getitem__(self, arr: Any) -> Any:
        _logger.debug("RLEArray.__getitem__(arr=%s(...))", type(arr).__name__)
        if isinstance(arr, tuple):
            # This is for example called by Pandas as values[:, None] to prepare the data for the cythonized
            # aggregation. Since we do not want to support the the aggregation over decompression, it is OK to not
            # implement this.
            raise NotImplementedError(
                "__getitem__ does currently only work w/ a single parameter"
            )

        if is_array_like(arr) or isinstance(arr, list):
            warnings.warn(
                "performance: __getitem__ with list is slow", PerformanceWarning
            )
            arr = _normalize_arraylike_indexing(arr, len(self))

            if arr.dtype == np.bool_:
                arr = np.arange(len(self))[arr]
            else:
                arr = arr.astype(int)

            arr[arr < 0] += len(self)

            result = np.asarray(
                [find_single_index(self._data, self._positions, i) for i in arr],
                dtype=self.dtype._dtype,
            )

            return self._from_sequence(result)
        elif isinstance(arr, slice):
            data, positions = find_slice(self._data, self._positions, arr)
            parent_normalized = NormalizedSlice.from_slice(
                get_len(self._projection.master.positions),
                self._projection.projection_slice,
            )
            child_normalized = NormalizedSlice.from_slice(len(self), arr)
            subslice = parent_normalized.project(child_normalized).to_slice()
            result = RLEArray(data=data, positions=positions)
            self._projection.master.register_change(result, subslice)
            return result
        else:
            if arr < 0:
                arr = arr + len(self)
            return find_single_index(self._data, self._positions, arr)

    def __setitem__(self, index: Any, data: Any) -> None:
        _logger.debug("RLEArray.__setitem__(...)")

        # get master data
        orig = decompress(
            data=self._projection.master.data,
            positions=self._projection.master.positions,
        )

        # get our view
        if self._projection.projection_slice is not None:
            sub = orig[self._projection.projection_slice]
        else:
            sub = orig

        # prepare index
        if is_array_like(index) or isinstance(index, list):
            index = _normalize_arraylike_indexing(index, len(self))

        # modify master data through view
        sub[index] = data

        # commit to all views (including self)
        data, positions = compress(orig)
        self._projection.master.modify(data, positions)

    def __len__(self) -> int:
        _logger.debug("RLEArray.__len__()")
        return get_len(self._positions)

    @property
    def dtype(self) -> RLEDtype:
        _logger.debug("RLEArray.dtype")
        return self._dtype

    @property
    def nbytes(self) -> int:
        _logger.debug("RLEArray.nbytes")
        return int(self._data.nbytes) + int(self._positions.nbytes)

    def isna(self) -> "RLEArray":
        _logger.debug("RLEArray.isna()")
        return RLEArray(data=pd.isna(self._data), positions=self._positions.copy())

    def take(
        self, indices: Sequence[int], allow_fill: bool = False, fill_value: Any = None
    ) -> "RLEArray":
        _logger.debug(
            "RLEArray.take(indices=%s(len=%s), allow_fill=%r, fill_value=%r)",
            type(indices).__name__,
            len(indices),
            allow_fill,
            fill_value,
        )
        if fill_value is None:
            fill_value = self.dtype.na_value

        indices = np.asarray(indices)

        data, positions = take(
            self._data, self._positions, indices, allow_fill, fill_value
        )
        return RLEArray(data=data, positions=positions)

    def copy(self) -> "RLEArray":
        _logger.debug("RLEArray.copy()")
        return RLEArray(data=self._data.copy(), positions=self._positions.copy())

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence["RLEArray"]) -> "RLEArray":
        t_to_concat = type(to_concat)
        to_concat = list(to_concat)
        _logger.debug(
            "RLEArray._concat_same_type(to_concat=%s(len=%i))",
            t_to_concat.__name__,
            len(to_concat),
        )
        data, positions = concat(
            [s._data for s in to_concat], [s._positions for s in to_concat]
        )
        return RLEArray(data=data, positions=positions)

    def __array__(self, dtype: Any = None) -> Any:
        _logger.debug("RLEArray.__array__(dtype=%r)", dtype)
        warnings.warn("performance: __array__ blows up data", PerformanceWarning)
        if dtype is None:
            dtype = self.dtype._dtype

        return decompress(self._data, self._positions, dtype)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        _logger.debug("RLEArray.astype(dtype=%r, copy=%r)", dtype, copy)
        if isinstance(dtype, RLEDtype):
            if (not copy) and (dtype == self.dtype):
                return self
            return RLEArray(
                data=self._data.astype(dtype._dtype), positions=self._positions.copy()
            )
        if isinstance(dtype, pd.StringDtype):
            # TODO: fast-path
            return StringArray._from_sequence([str(x) for x in self])
        return np.array(self, dtype=dtype, copy=copy)

    def _get_reduce_data(self, skipna: bool) -> Any:
        data = self._data
        if skipna:
            data = data[pd.notnull(data)]
        return data

    def _get_reduce_data_len(self, skipna: bool) -> Any:
        data = self._data
        lengths = self._lengths
        if skipna:
            mask = pd.notnull(data)
            data = data[mask]
            lengths = lengths[mask]
        return data, lengths

    def all(self, axis: Optional[int] = 0, out: Any = None) -> bool:
        _logger.debug("RLEArray.all()")
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        return bool(np.all(self._data))

    def any(self, axis: Optional[int] = 0, out: Any = None) -> bool:
        _logger.debug("RLEArray.any(axis=%r, out=%r)", axis, out)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        return bool(np.any(self._data))

    def kurt(self, skipna: bool = True) -> Any:
        _logger.debug("RLEArray.kurt(skipna=%r)", skipna)
        # TODO: fast-path
        data = np.asarray(self)
        return pd.Series(data).kurt(skipna=skipna)

    def max(self, skipna: bool = True, axis: Optional[int] = 0, out: Any = None) -> Any:
        _logger.debug("RLEArray.max(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        data = self._get_reduce_data(skipna)
        if len(data):
            return np.max(data)
        else:
            return self.dtype.na_value

    def mean(
        self,
        skipna: bool = True,
        dtype: Optional[Any] = None,
        axis: Optional[int] = 0,
        out: Any = None,
    ) -> Any:
        _logger.debug("RLEArray.mean(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")
        if dtype is not None:
            raise NotImplementedError("dtype parameter is not supported.")

        data, lengths = self._get_reduce_data_len(skipna)
        n = lengths.sum() if skipna else len(self)
        if n == 0:
            return self.dtype.na_value
        else:
            return np.dot(data, lengths) / np.float64(n)

    def median(
        self, skipna: bool = True, axis: Optional[int] = 0, out: Any = None
    ) -> Any:
        _logger.debug("RLEArray.median(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        # TODO: fast-path
        data = np.asarray(self)
        if skipna:
            data = data[pd.notnull(data)]
        return np.median(data)

    def min(self, skipna: bool = True, axis: Optional[int] = 0, out: Any = None) -> Any:
        _logger.debug("RLEArray.min(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        data = self._get_reduce_data(skipna)
        if len(data):
            return np.min(data)
        else:
            return self.dtype.na_value

    def prod(
        self, skipna: bool = True, axis: Optional[int] = 0, out: Any = None
    ) -> Any:
        _logger.debug("RLEArray.prod(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        data, lengths = self._get_reduce_data_len(skipna)
        return np.prod(np.power(data, lengths))

    def skew(self, skipna: bool = True) -> Any:
        _logger.debug("RLEArray.skew(skipna=%r)", skipna)
        # TODO: fast-path
        data = np.asarray(self)
        return pd.Series(data).skew(skipna=skipna)

    def std(
        self,
        skipna: bool = True,
        ddof: int = 1,
        dtype: Optional[Any] = None,
        axis: Optional[int] = 0,
        out: Any = None,
    ) -> Any:
        _logger.debug("RLEArray.std(skipna=%r, ddof=%r)", skipna, ddof)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")
        if dtype is not None:
            raise NotImplementedError("dtype parameter is not supported.")

        # TODO: fast-path
        data = np.asarray(self).astype(dtype)
        # use pandas-style std, since numpy results in different results
        return pd.Series(data).std(skipna=skipna, ddof=ddof)

    def sum(self, skipna: bool = True, axis: Optional[int] = 0, out: Any = None) -> Any:
        _logger.debug("RLEArray.sum(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")

        data, lengths = self._get_reduce_data_len(skipna)
        return np.dot(data, lengths)

    def var(
        self,
        skipna: bool = True,
        ddof: int = 1,
        dtype: Optional[Any] = None,
        axis: Optional[int] = 0,
        out: Any = None,
    ) -> Any:
        _logger.debug("RLEArray.var(skipna=%r)", skipna)
        if (axis is not None) and (axis != 0):
            raise NotImplementedError("Only axis=0 is supported.")
        if out is not None:
            raise NotImplementedError("out parameter is not supported.")
        if dtype is not None:
            raise NotImplementedError("dtype parameter is not supported.")

        # TODO: fast-path
        data = np.asarray(self).astype(dtype)
        # use pandas-style var, since numpy results in different results
        return pd.Series(data).var(skipna=skipna, ddof=ddof)

    def _reduce(self, name: str, skipna: bool = True, **kwargs: Any) -> Any:
        _logger.debug(
            "RLEArray._reduce(name=%r, skipna=%r, **kwargs=%r)", name, skipna, kwargs
        )
        if name == "all":
            return self.all()
        elif name == "any":
            return self.any()
        elif name == "kurt":
            return self.kurt(skipna=skipna)
        elif name == "max":
            return self.max(skipna=skipna)
        elif name == "mean":
            return self.mean(skipna=skipna)
        elif name == "median":
            return self.median(skipna=skipna)
        elif name == "min":
            return self.min(skipna=skipna)
        elif name == "prod":
            return self.prod(skipna=skipna)
        elif name == "skew":
            return self.skew(skipna=skipna)
        elif name == "std":
            return self.std(skipna=skipna, ddof=int(kwargs.get("ddof", 1)))
        elif name == "sum":
            return self.sum(skipna=skipna)
        elif name == "var":
            return self.var(skipna=skipna)
        else:
            raise NotImplementedError(f"reduction {name} is not implemented.")

    def view(self, dtype: Optional[Any] = None) -> Any:
        _logger.debug("RLEArray.view(dtype=%r)", dtype)
        if dtype is None:
            dtype = self.dtype._dtype
        if isinstance(dtype, RLEDtype):
            dtype = dtype._dtype
        if dtype != self.dtype._dtype:
            raise ValueError("Cannot create view with different dtype.")

        result = RLEArray(data=self._data.copy(), positions=self._positions.copy())
        self._projection.master.register_change(result, None)
        return result

    def dropna(self) -> "RLEArray":
        _logger.debug("RLEArray.dropna()")
        data, positions = dropna(self._data, self._positions)
        return RLEArray(data=data, positions=positions)

    def value_counts(self, dropna: bool = True) -> pd.Series:
        _logger.debug("RLEArray.value_counts(dropna=%r)", dropna)
        # TODO: add fast-path
        return pd.Series(np.asarray(self)).value_counts(dropna=dropna)

    def __iter__(self) -> Iterator[Any]:
        _logger.debug("RLEArray.__iter__()")
        warnings.warn("performance: __iter__ blows up entire data", PerformanceWarning)
        return gen_iterator(self._data, self._positions)

    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Union[None, "RLEArray", np.ndarray]:
        _logger.debug("RLEArray.__array_ufunc__(...)")
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (RLEArray,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(np.asarray(x) if isinstance(x, RLEArray) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                np.asarray(x) if isinstance(x, RLEArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if out:
            for x, y in zip(out, kwargs["out"]):
                if isinstance(x, RLEArray):
                    x[:] = y

        def maybe_from_sequence(x: np.ndarray) -> Union[RLEArray, np.ndarray]:
            if x.ndim == 1:
                # suitable for RLE compression
                return type(self)._from_sequence(x)
            else:
                # likely a broadcast operation
                return x

        if type(result) is tuple:
            # multiple return values
            return tuple(maybe_from_sequence(x) for x in result)
        elif method == "at":
            assert result is None

            # inplace modification
            self[:] = inputs[0]

            # no return value
            return None
        else:
            # one return value
            return maybe_from_sequence(result)

    def __eq__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.eq)

    def __ne__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.ne)

    def __gt__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.gt)

    def __ge__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.ge)

    def __lt__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.lt)

    def __le__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.le)

    def __add__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.add)

    def __radd__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=ops.radd)

    def __sub__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.sub)

    def __mul__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.mul)

    def __rmul__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=rev(operator.mul))

    def __truediv__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.truediv)

    def __floordiv__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.floordiv)

    def __mod__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.mod)

    def __pow__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.pow)

    def __and__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.and_)

    def __or__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.or_)

    def __xor__(self, other: Any) -> Union["RLEArray", np.ndarray]:
        return self._apply_binary_operator(other, op=operator.xor)

    def __pos__(self) -> "RLEArray":
        return self._apply_unary_operator(op=operator.pos)

    def __neg__(self) -> "RLEArray":
        return self._apply_unary_operator(op=operator.neg)

    def __abs__(self) -> "RLEArray":
        return self._apply_unary_operator(op=operator.abs)

    def __invert__(self) -> "RLEArray":
        _logger.debug("RLEArray.__invert__()")
        return self._apply_unary_operator(op=operator.inv)

    def _apply_binary_operator(
        self, other: Any, op: Any
    ) -> Union["RLEArray", np.ndarray]:
        if isinstance(other, (ABCSeries, ABCIndexClass)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        if is_scalar(other):
            with np.errstate(invalid="ignore"):
                new_data = op(self._data, other)
            return RLEArray(*recompress(new_data, self._positions))
        elif isinstance(other, RLEArray):
            if len(self) != len(other):
                raise ValueError("arrays have different lengths")
            extended_positions = extend_positions(self._positions, other._positions)
            data_self = extend_data(
                data=self._data,
                positions=self._positions,
                extended_positions=extended_positions,
            )
            data_other = extend_data(
                data=other._data,
                positions=other._positions,
                extended_positions=extended_positions,
            )
            with np.errstate(invalid="ignore"):
                new_data = op(data_self, data_other)
            return RLEArray(*recompress(new_data, extended_positions))
        else:
            array = self.__array__()
            with np.errstate(invalid="ignore"):
                return op(array, other)

    def _apply_unary_operator(self, op: Any) -> "RLEArray":
        return RLEArray(data=op(self._data), positions=self._positions.copy())

    def shift(self, periods: int = 1, fill_value: object = None) -> "RLEArray":
        self2 = self
        dtype = self.dtype

        if isna(fill_value):
            fill_value = self.dtype.na_value
            np_dtype_fill = np.asarray([fill_value]).dtype
            if np_dtype_fill.kind != dtype.kind:
                dtype = RLEDtype(np_dtype_fill)
                self2 = self.astype(dtype)

        if not len(self) or periods == 0:
            return self2.copy()

        empty = RLEArray(
            data=np.asarray([fill_value], dtype=dtype._dtype),
            positions=np.asarray([min(abs(periods), len(self))], dtype=POSITIONS_DTYPE),
        )

        if periods > 0:
            a = empty
            b = self2[:-periods]
        else:
            a = self2[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def fillna(
        self,
        value: Any = None,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Any:
        # TODO: fast-path
        arr = pd.Series(np.asarray(self)).array.fillna(value, method, limit).to_numpy()
        data, positions = compress(arr)
        return RLEArray(data=data, positions=positions)
