"""
Helpers that allows us to deal with Python slicing.

The issue with Python slicing are:

- ``slice`` type:
  - the types in ``slice`` are completely unchecked (can even be a string or any user-provided type)
  - the consistency of the values in ``slice`` are unchecked
  - there is not information about the container size (which makes consistency checks more complicated)

- ``slice.step`` value:
  - ``step`` has the implicit default 1
  - there can be forward and backward slices depending on the ``step`` value
  - there can be step sizes which are not modulo 1

- ``slice.start`` and ``slice.stop`` values:
  - the implicit defaults of ``start`` and ``stop`` depend on ``step`` (is it positive or negative?)
  - ``start`` and ``stop`` can be negative (aka "from the end")
  - ``start`` and ``stop`` can over/underflow the container

We do not want to deal with all these edge cases in every code snipped that deals with slicing, so we introduce
:class:`NormalizedSlice` that solves the issue in a central place.
"""
from typing import Optional

import numpy as np


class NormalizedSlice:
    """
    A normalized slice.

    .. important::

        Do not try to construct this class by hand. Use :func:`NormalizedSlice.from_slice` instead!

    Parameters
    ----------
    start
        First absolute index in the container (inclusive start). Always positive.
    stop
        Last absolute index not being part of the slice (exclusive end). Counted from the container start. Can be
        negative for refersed slices (aka ``step < 0``). Must be normalized so that ``abs(stop - start) % step == 0``.
        For forward slices (``step > 0``), this must be greater than ``start``. For backward slices (``step < 0``) this
        must be less than ``start``. For empty slices (``start = stop``), ``start``, ``stop`` and ``step`` have the
        fixed values 0, 0 and 1.
    step
        Step size. Must not be ``0``.
    container_length
        Size of the container this slice applies to. Must not be negative. For empty containers
        (``container_length = 0``), ``start``, ``stop`` and ``step`` have the fixed values 0, 0 and 1.
    """

    def __init__(self, start: int, stop: int, step: int, container_length: int):
        if not isinstance(start, int):
            raise TypeError(f"start must be int but is {type(start).__name__}")
        if not isinstance(stop, int):
            raise TypeError(f"stop must be int but is {type(stop).__name__}")
        if not isinstance(step, int):
            raise TypeError(f"step must be int but is {type(step).__name__}")
        if not isinstance(container_length, int):
            raise TypeError(
                f"container_length must be int but is {type(container_length).__name__}"
            )

        self._start = start
        self._stop = stop
        self._step = step
        self._container_length = container_length

        self._verify()

    def _verify(self) -> None:
        """
        Verify integrity.
        """
        if self.container_length < 0:
            raise ValueError(
                f"container_length ({self.container_length}) must be greater or equal to zero"
            )
        elif self.container_length == 0:
            self._verify_container_empty()
        else:
            self._verify_container_not_empty()

    def _verify_container_empty(self) -> None:
        """
        Verify integrity in case the container is empty (``container_length = 0``).
        """
        # empty container => special values required
        if self.start != 0:
            raise ValueError(
                f"for empty containers, start must be 0 but is {self.start}"
            )

        if self.stop != 0:
            raise ValueError(f"for empty containers, stop must be 0 but is {self.stop}")

        if self.step != 1:
            raise ValueError(f"for empty containers, step must be 1 but is {self.step}")

    def _verify_container_not_empty(self) -> None:
        """
        Verify integrity in case the container is not empty (``container_length > 0``).
        """
        if (self.start < 0) or (self.start >= self.container_length):
            raise ValueError(
                f"start ({self.start}) must be in [0,{self.container_length}) but is not"
            )

        if (self.stop < -abs(self.step)) or (
            self.stop >= self.container_length + abs(self.step)
        ):
            raise ValueError(
                f"stop ({self.stop}) must be in [{-abs(self.step)},{self.container_length + abs(self.step)}) but is not"
            )

        if self.start == self.stop:
            # empty slice
            if self.start != 0:
                raise ValueError(
                    f"for empty slices, start and stop must be 0 but are {self.start}"
                )
            if self.step != 1:
                raise ValueError(f"for empty slices, step must be 1 but is {self.step}")
        else:
            # non-empty slice
            if self.step == 0:
                raise ValueError("step cannot be zero")
            elif self.step > 0:
                # forward slice
                if self.start > self.stop:
                    raise ValueError(
                        "for forward slices, stop must be greater or equal to start"
                    )
            else:
                # backward slice
                if self.stop > self.start:
                    raise ValueError(
                        "for backward slices, start must be greater or equal to stop"
                    )

            if abs(self.start - self.stop) % abs(self.step) != 0:
                raise ValueError("start->stop distance is not modulo step")

    @property
    def start(self) -> int:
        """
        Start index of the slice. Inclusive start.
        """
        return self._start

    @property
    def stop(self) -> int:
        """
        Stop index of the slice. Exclusive end.
        """
        return self._stop

    @property
    def step(self) -> int:
        """
        Step width.
        """
        return self._step

    @property
    def container_length(self) -> int:
        """
        Length of the container.
        """
        return self._container_length

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(start={self.start}, stop={self.stop}, step={self.step}, container_length="
            f"{self.container_length})"
        )

    def __len__(self) -> int:
        return self._calc_len(start=self.start, stop=self.stop, step=self.step)

    @classmethod
    def _calc_len(cls, start: int, stop: int, step: int) -> int:
        """
        Calculate slice length.

        Parameters
        ----------
        start
            Inclusive start index.
        stop
            Exclusive stop index.
        step
            Step width.
        """
        delta = abs(stop - start)
        steps = delta // abs(step)
        if delta % abs(step) != 0:
            steps += 1
        return steps

    @classmethod
    def _check_and_prepare_slice(cls, s: Optional[slice]) -> slice:
        """
        Check and prepare input slice for convertion.
        """
        if s is None:
            s = slice(None, None, None)

        if not isinstance(s, slice):
            raise TypeError(f"slice must be a slice but is {type(s).__name__}")

        if (s.start is not None) and not isinstance(s.start, (int, np.int64)):
            raise TypeError(
                f"slice start must be int or None but is {type(s.start).__name__}"
            )
        start = None if s.start is None else int(s.start)

        if (s.stop is not None) and not isinstance(s.stop, (int, np.int64)):
            raise TypeError(
                f"slice stop must be int or None but is {type(s.stop).__name__}"
            )
        stop = None if s.stop is None else int(s.stop)

        if (s.step is not None) and not isinstance(s.step, (int, np.int64)):
            raise TypeError(
                f"slice step must be int or None but is {type(s.step).__name__}"
            )
        if s.step == 0:
            raise ValueError("slice step cannot be zero")
        step = None if s.step is None else int(s.step)

        return slice(start, stop, step)

    @classmethod
    def from_slice(cls, container_length: int, s: Optional[slice]) -> "NormalizedSlice":
        """
        Create a new :class:`NormalizedSlice` from a given Python ``slice`` and container length.

        Parameters
        ----------
        container_length
            Non-negative container length.
        s
            Slice or ``None`` (for "take all").

        Raises
        ------
        TypeError: If ``s`` is not ``None`` and not a ``slice`` or any of the arguments for ``slice`` are neither
                   ``None`` nor an integer.
        ValueError: Illegal ``slice`` values or ``container_length``.
        """
        s2 = cls._check_and_prepare_slice(s)

        if not isinstance(container_length, (int, np.int64)):
            raise TypeError(
                f"container_length must be an int but is {type(container_length).__name__}"
            )
        if container_length < 0:
            raise ValueError("container_length cannot be negative")

        if container_length == 0:
            return cls(start=0, stop=0, step=1, container_length=0)

        container_length = int(container_length)

        default_start, default_stop = 0, container_length

        if s2.step is not None:
            step = s2.step
            if step < 0:
                default_start, default_stop = default_stop - 1, default_start - 1
        else:
            step = 1

        def limit(x: int) -> int:
            a = min(default_start, default_stop)
            b = max(default_start, default_stop)
            return max(a, min(b, x))

        if s2.start is not None:
            if s2.start < 0:
                start = limit(container_length + s2.start)
            else:
                start = limit(s2.start)
        else:
            start = default_start

        if s2.stop is not None:
            if s2.stop < 0:
                stop = limit(container_length + s2.stop)
            else:
                stop = limit(s2.stop)
        else:
            stop = default_stop

        if step > 0:
            if stop < start:
                stop = start
        else:
            if stop > start:
                stop = start

        if start == stop:
            return cls(start=0, stop=0, step=1, container_length=container_length)

        # re-adjusting the range to be modulo `step`
        stop = start + step * cls._calc_len(start=start, stop=stop, step=step)

        return cls(start=start, stop=stop, step=step, container_length=container_length)

    def project(self, child: "NormalizedSlice") -> "NormalizedSlice":
        """
        Project a slice.

        Given a parent slice (``self``) which is applied first, calculate slice of this slice would look like so it can
        be applied to the original data.

        Parameters
        ----------
        child
            Second slice to apply.

        Raises
        ------
        TypeError: If ``child`` is not a ``NormalizedSlice``.
        ValueError: If ``child.container_length`` is not the length of ``self``.

        Example
        -------
        >>> # given some unknown data:
        >>> data = list(range(100))

        >>> # and two slices:
        >>> parent = slice(10, -8, 2)
        >>> child = slice(-20, -1, -1)

        >>> # and the application of both slices
        >>> expected = data[parent][child]

        >>> # construct a slice that does both steps at once
        >>> from rle_array._algorithms import NormalizedSlice
        >>> parent_normalized = NormalizedSlice.from_slice(len(data), parent)
        >>> child_normalized = NormalizedSlice.from_slice(len(parent), child)
        >>> projected = parent_normalized.project(child_normalized).to_slice()
        >>> actual = data[projected]
        >>> assert actual == expected
        """
        if not isinstance(child, NormalizedSlice):
            raise TypeError(
                f"child must be NormalizedSlice but is {type(child).__name__}"
            )
        if child.container_length != len(self):
            raise ValueError(
                f"container_length of child ({child.container_length}) must be length of parent ({len(self)})"
            )

        start = self.start + child.start * self.step
        stop = self.start + child.stop * self.step
        step = self.step * child.step

        return type(self)(
            start=start, stop=stop, step=step, container_length=self.container_length
        )

    def to_slice(self) -> Optional[slice]:
        """
        Convert :class:`NormalizedSlice` back to a slice.

        Returns ``None`` if no slicing is applied (e.g. the whole container with ``step=1`` is taken).
        """
        start: Optional[int] = self.start
        stop: Optional[int] = self.stop
        step: Optional[int] = self.step

        if self.step > 0:
            # forwards
            if self.start <= 0:
                start = None
            if self.stop >= self.container_length:
                stop = None
            if self.step == 1:
                step = None
        else:
            # backward
            if self.start >= self.container_length - 1:
                start = None
            if self.stop < 0:
                stop = None

        if (start is None) and (stop is None) and (step is None):
            return None
        else:
            return slice(start, stop, step)
