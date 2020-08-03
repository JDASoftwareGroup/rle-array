from typing import Optional, cast

import numpy as np
import pytest

from rle_array._slicing import NormalizedSlice


class TestConstructor:
    def test_ok_simple(self) -> None:
        s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
        assert s.start == 1
        assert s.stop == 11
        assert s.step == 2
        assert s.container_length == 100

    def test_ok_start_at_zero(self) -> None:
        NormalizedSlice(start=0, stop=10, step=2, container_length=100)

    def test_ok_stop_at_modulo_end(self) -> None:
        NormalizedSlice(start=0, stop=12, step=3, container_length=10)

    def test_ok_stop_at_modulo_begin(self) -> None:
        NormalizedSlice(start=0, stop=-3, step=-3, container_length=10)

    def test_ok_zero_length(self) -> None:
        NormalizedSlice(start=0, stop=0, step=1, container_length=0)

    def test_fail_start_none(self) -> None:
        with pytest.raises(TypeError, match="start must be int but is None"):
            NormalizedSlice(
                start=cast(int, None), stop=10, step=2, container_length=100
            )

    def test_fail_stop_none(self) -> None:
        with pytest.raises(TypeError, match="stop must be int but is None"):
            NormalizedSlice(start=1, stop=cast(int, None), step=2, container_length=100)

    def test_fail_step_none(self) -> None:
        with pytest.raises(TypeError, match="step must be int but is None"):
            NormalizedSlice(
                start=1, stop=10, step=cast(int, None), container_length=100
            )

    def test_fail_container_length_none(self) -> None:
        with pytest.raises(TypeError, match="container_length must be int but is None"):
            NormalizedSlice(start=1, stop=10, step=2, container_length=cast(int, None))

    def test_fail_step_zero(self) -> None:
        with pytest.raises(ValueError, match="step cannot be zero"):
            NormalizedSlice(start=1, stop=10, step=0, container_length=100)

    def test_fail_start_negative(self) -> None:
        with pytest.raises(
            ValueError, match=r"start \(-1\) must be in \[0,100\) but is not"
        ):
            NormalizedSlice(start=-1, stop=10, step=1, container_length=100)

    def test_fail_start_large(self) -> None:
        with pytest.raises(
            ValueError, match=r"start \(100\) must be in \[0,100\) but is not"
        ):
            NormalizedSlice(start=100, stop=10, step=1, container_length=100)

    def test_fail_stop_small(self) -> None:
        with pytest.raises(
            ValueError, match=r"stop \(-2\) must be in \[-1,101\) but is not"
        ):
            NormalizedSlice(start=2, stop=-2, step=-1, container_length=100)

    def test_fail_stop_large(self) -> None:
        with pytest.raises(
            ValueError, match=r"stop \(102\) must be in \[-1,101\) but is not"
        ):
            NormalizedSlice(start=2, stop=102, step=1, container_length=100)

    def test_fail_container_length_negative(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"container_length \(-1\) must be greater or equal to zero",
        ):
            NormalizedSlice(start=2, stop=102, step=1, container_length=-1)

    def test_fail_container_empty_start_fail(self) -> None:
        with pytest.raises(
            ValueError, match="for empty containers, start must be 0 but is 1"
        ):
            NormalizedSlice(start=1, stop=0, step=1, container_length=0)

    def test_fail_container_empty_stop_fail(self) -> None:
        with pytest.raises(
            ValueError, match="for empty containers, stop must be 0 but is 1"
        ):
            NormalizedSlice(start=0, stop=1, step=1, container_length=0)

    def test_fail_container_empty_step_fail(self) -> None:
        with pytest.raises(
            ValueError, match="for empty containers, step must be 1 but is 2"
        ):
            NormalizedSlice(start=0, stop=0, step=2, container_length=0)

    def test_fail_forward_slice_not_forward(self) -> None:
        with pytest.raises(
            ValueError,
            match="for forward slices, stop must be greater or equal to start",
        ):
            NormalizedSlice(start=1, stop=0, step=1, container_length=100)

    def test_fail_backward_slice_not_backward(self) -> None:
        with pytest.raises(
            ValueError,
            match="for backward slices, start must be greater or equal to stop",
        ):
            NormalizedSlice(start=0, stop=1, step=-1, container_length=100)

    def test_fail_slice_empty_start(self) -> None:
        with pytest.raises(
            ValueError, match="for empty slices, start and stop must be 0 but are 1"
        ):
            NormalizedSlice(start=1, stop=1, step=1, container_length=100)

    def test_fail_slice_empty_step(self) -> None:
        with pytest.raises(
            ValueError, match="for empty slices, step must be 1 but is 2"
        ):
            NormalizedSlice(start=0, stop=0, step=2, container_length=100)

    def test_fail_distance_not_modulo(self) -> None:
        with pytest.raises(
            ValueError,
            match="The distance between start and stop most be divisible by the step size",
        ):
            NormalizedSlice(start=0, stop=10, step=3, container_length=100)


class TestFrozen:
    def test_start(self) -> None:
        s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
        with pytest.raises(AttributeError, match="can't set attribute"):
            s.start = 2  # type: ignore

    def test_stop(self) -> None:
        s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
        with pytest.raises(AttributeError, match="can't set attribute"):
            s.stop = 2  # type: ignore

    def test_step(self) -> None:
        s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
        with pytest.raises(AttributeError, match="can't set attribute"):
            s.step = 3  # type: ignore

    def test_container_length(self) -> None:
        s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
        with pytest.raises(AttributeError, match="can't set attribute"):
            s.container_length = 3  # type: ignore


def test_repr() -> None:
    s = NormalizedSlice(start=1, stop=11, step=2, container_length=100)
    assert repr(s) == "NormalizedSlice(start=1, stop=11, step=2, container_length=100)"


@pytest.mark.parametrize(
    "s, expected",
    [
        (  # empty
            # s
            NormalizedSlice(start=0, stop=0, step=1, container_length=0),
            # expected
            0,
        ),
        (  # simple, forward
            # s
            NormalizedSlice(start=0, stop=10, step=1, container_length=100),
            # expected
            10,
        ),
        (  # simple, backward
            # s
            NormalizedSlice(start=9, stop=-1, step=-1, container_length=100),
            # expected
            10,
        ),
        (  # even, forward
            # s
            NormalizedSlice(start=0, stop=10, step=2, container_length=100),
            # expected
            5,
        ),
        (  # even, backward
            # s
            NormalizedSlice(start=9, stop=-1, step=-2, container_length=100),
            # expected
            5,
        ),
        (  # complex, forward
            # s
            NormalizedSlice(start=10, stop=22, step=3, container_length=100),
            # expected
            4,
        ),
        (  # complex, backward
            # s
            NormalizedSlice(start=19, stop=7, step=-3, container_length=100),
            # expected
            4,
        ),
    ],
)
def test_len(s: NormalizedSlice, expected: int) -> None:
    assert len(s) == expected


class TestFromSlice:
    def test_fail_slice_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="slice must be a slice but is str"):
            NormalizedSlice.from_slice(container_length=10, s=cast(slice, "foo"))

    def test_fail_slice_start_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match="slice start must be int or None but is str"
        ):
            NormalizedSlice.from_slice(container_length=10, s=slice("foo", 20, 2))

    def test_fail_slice_stop_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match="slice stop must be int or None but is str"
        ):
            NormalizedSlice.from_slice(container_length=10, s=slice(2, "foo", 2))

    def test_fail_slice_step_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match="slice step must be int or None but is str"
        ):
            NormalizedSlice.from_slice(container_length=10, s=slice(2, 20, "foo"))

    def test_fail_step_zero(self) -> None:
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            NormalizedSlice.from_slice(container_length=10, s=slice(2, 10, 0))

    def test_fail_container_length_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match="container_length must be an int but is str"
        ):
            NormalizedSlice.from_slice(
                container_length=cast(int, "foo"), s=slice(2, 10, 2)
            )

    def test_fail_container_length_negative(self) -> None:
        with pytest.raises(ValueError, match="container_length cannot be negative"):
            NormalizedSlice.from_slice(container_length=-1, s=slice(2, 10, 2))

    @pytest.mark.parametrize(
        "container_length, s, expected",
        [
            (  # empty
                # container_length
                0,
                # s
                None,
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=0),
            ),
            (  # implicit full via None
                # container_length
                100,
                # s
                None,
                # expected
                NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            ),
            (  # explicit full via slice
                # container_length
                100,
                # s
                slice(None, None, None),
                # expected
                NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            ),
            (  # explicit full
                # container_length
                100,
                # s
                slice(0, 100, 1),
                # expected
                NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            ),
            (  # full reverse
                # container_length
                100,
                # s
                slice(None, None, -1),
                # expected
                NormalizedSlice(start=99, stop=-1, step=-1, container_length=100),
            ),
            (  # start negative
                # container_length
                100,
                # s
                slice(-20, None, None),
                # expected
                NormalizedSlice(start=80, stop=100, step=1, container_length=100),
            ),
            (  # start negative overflow container
                # container_length
                100,
                # s
                slice(-1000, None, None),
                # expected
                NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            ),
            (  # stop negative
                # container_length
                100,
                # s
                slice(None, -20, None),
                # expected
                NormalizedSlice(start=0, stop=80, step=1, container_length=100),
            ),
            (  # stop negative overflow container
                # container_length
                100,
                # s
                slice(None, -1000, None),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=100),
            ),
            (  # stop negative overflow start
                # container_length
                100,
                # s
                slice(10, -1000, None),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=100),
            ),
            (  # stop negative overflow start reverse
                # container_length
                100,
                # s
                slice(10, -10, -1),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=100),
            ),
            (  # modulo normlization forward
                # container_length
                10,
                # s
                slice(0, 10, 3),
                # expected
                NormalizedSlice(start=0, stop=12, step=3, container_length=10),
            ),
            (  # modulo normlization forward, empty
                # container_length
                10,
                # s
                slice(0, 0, 3),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=10),
            ),
            (  # modulo normlization backward
                # container_length
                10,
                # s
                slice(0, -1000, -3),
                # expected
                NormalizedSlice(start=0, stop=-3, step=-3, container_length=10),
            ),
            (  # modulo normlization backward, empty
                # container_length
                10,
                # s
                slice(0, 0, -3),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=10),
            ),
            (  # numpy.int64
                # container_length
                np.int64(100),
                # s
                slice(np.int64(0), np.int64(100), np.int64(1)),
                # expected
                NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            ),
        ],
    )
    def test_ok(
        self, container_length: int, s: Optional[slice], expected: NormalizedSlice
    ) -> None:
        actual = NormalizedSlice.from_slice(container_length, s)
        assert type(actual) == NormalizedSlice
        assert actual.start == expected.start
        assert type(actual.start) == int
        assert actual.stop == expected.stop
        assert type(actual.stop) == int
        assert actual.step == expected.step
        assert type(actual.step) == int
        assert actual.container_length == expected.container_length
        assert type(actual.container_length) == int


class TestProject:
    def test_fail_no_normalizedslice(self) -> None:
        s1 = NormalizedSlice(start=0, stop=10, step=1, container_length=100)
        s2 = slice(1, 2, 1)
        with pytest.raises(
            TypeError, match="child must be NormalizedSlice but is slice"
        ):
            s1.project(cast(NormalizedSlice, s2))

    def test_fail_len_diff(self) -> None:
        s1 = NormalizedSlice(start=0, stop=10, step=1, container_length=100)
        s2 = NormalizedSlice(start=0, stop=10, step=1, container_length=20)
        with pytest.raises(
            ValueError,
            match=r"container_length of child \(20\) must be length of parent \(10\)",
        ):
            s1.project(s2)

    @pytest.mark.parametrize(
        "s1, s2, expected",
        [
            (  # simple full take
                # s1
                NormalizedSlice(start=0, stop=10, step=1, container_length=100),
                # s2
                NormalizedSlice(start=0, stop=10, step=1, container_length=10),
                # expected
                NormalizedSlice(start=0, stop=10, step=1, container_length=100),
            ),
            (  # reverse reverse
                # s1
                NormalizedSlice(start=9, stop=-1, step=-1, container_length=100),
                # s2
                NormalizedSlice(start=9, stop=-1, step=-1, container_length=10),
                # expected
                NormalizedSlice(start=0, stop=10, step=1, container_length=100),
            ),
            (  # two modulos
                # s1
                NormalizedSlice(start=2, stop=29, step=3, container_length=100),
                # s2
                NormalizedSlice(start=1, stop=7, step=3, container_length=9),
                # expected
                NormalizedSlice(start=5, stop=23, step=9, container_length=100),
            ),
            (  # take empty
                # s1
                NormalizedSlice(start=1, stop=9, step=2, container_length=100),
                # s2
                NormalizedSlice(start=0, stop=0, step=1, container_length=4),
                # expected
                NormalizedSlice(start=0, stop=0, step=1, container_length=100),
            ),
        ],
    )
    def test_ok(
        self, s1: NormalizedSlice, s2: NormalizedSlice, expected: NormalizedSlice
    ) -> None:
        actual = s1.project(s2)
        assert type(actual) == NormalizedSlice
        assert actual.start == expected.start
        assert actual.stop == expected.stop
        assert actual.step == expected.step
        assert actual.container_length == expected.container_length


@pytest.mark.parametrize(
    "s, expected",
    [
        (  # full take
            # s
            NormalizedSlice(start=0, stop=100, step=1, container_length=100),
            # expected
            None,
        ),
        (  # full reverse
            # s
            NormalizedSlice(start=99, stop=-1, step=-1, container_length=100),
            # expected
            slice(None, None, -1),
        ),
        (  # only start
            # s
            NormalizedSlice(start=1, stop=100, step=1, container_length=100),
            # expected
            slice(1, None, None),
        ),
        (  # only stop
            # s
            NormalizedSlice(start=0, stop=99, step=1, container_length=100),
            # expected
            slice(None, 99, None),
        ),
        (  # only step
            # s
            NormalizedSlice(start=0, stop=100, step=2, container_length=100),
            # expected
            slice(None, None, 2),
        ),
        (  # complex
            # s
            NormalizedSlice(start=1, stop=22, step=3, container_length=100),
            # expected
            slice(1, 22, 3),
        ),
    ],
)
def test_to_slice(s: NormalizedSlice, expected: Optional[slice]) -> None:
    actual = s.to_slice()
    if expected is None:
        assert actual is None
    else:
        assert isinstance(actual, slice)
        assert type(actual) == slice
        assert actual.start == expected.start
        assert actual.stop == expected.stop
        assert actual.step == expected.step
