from typing import Any, Callable, List, Optional, cast

import numpy as np
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.cast import find_common_type

import rle_array


@register_extension_dtype
class RLEDtype(ExtensionDtype):
    _metadata = ("_dtype",)

    def __init__(self, dtype: Any):
        self._dtype = np.dtype(dtype)

    @property
    def type(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self._dtype.type)

    @property
    def kind(self) -> str:
        return str(self._dtype.kind)

    @property
    def name(self) -> str:
        return f"RLEDtype[{self._dtype}]"

    @classmethod
    def construct_from_string(cls, string: str) -> "RLEDtype":
        """
        Strict construction from a string, raise a TypeError if not possible.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        prefix = "RLEDtype["
        suffix = "]"
        if not (string.startswith(prefix) and string.endswith(suffix)):
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        sub = string[len(prefix) : -len(suffix)]
        return cls(np.dtype(sub))

    @classmethod
    def construct_array_type(
        cls,
    ) -> Callable[[np.ndarray, np.ndarray], "rle_array.RLEArray"]:
        return rle_array.RLEArray

    @property
    def _is_numeric(self) -> bool:
        # exclude object, str, unicode, void.
        return self.kind in set("biufc")

    @property
    def _is_boolean(self) -> bool:
        return self.kind == "b"

    def _get_common_dtype(self, dtypes: List[Any]) -> Optional[Any]:
        unpacked_dtypes = []
        only_rle = True
        for t in dtypes:
            if isinstance(t, RLEDtype):
                unpacked_dtypes.append(t._dtype)
            else:
                unpacked_dtypes.append(t)
                only_rle = False

        # ask pandas for help
        suggested_type = find_common_type(unpacked_dtypes)

        # prefer RLE
        if (suggested_type is not None) and only_rle:
            return RLEDtype(suggested_type)
        else:
            return suggested_type

    def __repr__(self) -> str:
        return f"RLEDtype({self._dtype!r})"
