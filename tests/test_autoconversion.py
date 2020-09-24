from typing import Optional

import numpy as np
import pandas as pd
import pytest
from pandas import testing as pdt

from rle_array.autoconversion import auto_convert_to_rle
from rle_array.dtype import RLEDtype

pytestmark = pytest.mark.filterwarnings("ignore:performance")


@pytest.mark.parametrize(
    "orig, threshold, expected",
    [
        (
            # orig
            pd.DataFrame(
                {
                    "int64": pd.Series([1], dtype=np.int64),
                    "int32": pd.Series([1], dtype=np.int32),
                    "uint64": pd.Series([1], dtype=np.uint64),
                    "float64": pd.Series([1.2], dtype=np.float64),
                    "bool": pd.Series([True], dtype=np.bool_),
                    "object": pd.Series(["foo"], dtype=np.object_),
                    "datetime64": pd.Series(
                        [pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"
                    ),
                }
            ),
            # threshold
            None,
            # expected
            pd.DataFrame(
                {
                    "int64": pd.Series([1], dtype=RLEDtype(np.int64)),
                    "int32": pd.Series([1], dtype=RLEDtype(np.int32)),
                    "uint64": pd.Series([1], dtype=RLEDtype(np.uint64)),
                    "float64": pd.Series([1.2], dtype=RLEDtype(np.float64)),
                    "bool": pd.Series([True], dtype=RLEDtype(np.bool_)),
                    "object": pd.Series(["foo"]).astype(RLEDtype(np.object_)),
                    "datetime64": pd.Series(
                        [pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"
                    ),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "int64": pd.Series([1], dtype=np.int64),
                    "int32": pd.Series([1], dtype=np.int32),
                    "uint64": pd.Series([1], dtype=np.uint64),
                    "float64": pd.Series([1.2], dtype=np.float64),
                    "bool": pd.Series([True], dtype=np.bool_),
                    "object": pd.Series(["foo"], dtype=np.object_),
                    "datetime64": pd.Series(
                        [pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"
                    ),
                }
            ),
            # threshold
            2.0,
            # expected
            pd.DataFrame(
                {
                    "int64": pd.Series([1], dtype=RLEDtype(np.int64)),
                    "int32": pd.Series([1], dtype=np.int32),
                    "uint64": pd.Series([1], dtype=RLEDtype(np.uint64)),
                    "float64": pd.Series([1.2], dtype=RLEDtype(np.float64)),
                    "bool": pd.Series([True], dtype=np.bool_),
                    "object": pd.Series(["foo"]).astype(RLEDtype(np.object_)),
                    "datetime64": pd.Series(
                        [pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"
                    ),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
            # threshold
            None,
            # expected
            pd.DataFrame(
                {
                    "single_value": pd.Series(
                        [1, 1, 1, 1, 1, 1], dtype=RLEDtype(np.int64)
                    ),
                    "two_values": pd.Series(
                        [1, 1, 1, 2, 2, 2], dtype=RLEDtype(np.int64)
                    ),
                    "increasing": pd.Series(
                        [1, 2, 3, 4, 5, 6], dtype=RLEDtype(np.int64)
                    ),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
            # threshold
            0.9,
            # expected
            pd.DataFrame(
                {
                    "single_value": pd.Series(
                        [1, 1, 1, 1, 1, 1], dtype=RLEDtype(np.int64)
                    ),
                    "two_values": pd.Series(
                        [1, 1, 1, 2, 2, 2], dtype=RLEDtype(np.int64)
                    ),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
            # threshold
            0.5,
            # expected
            pd.DataFrame(
                {
                    "single_value": pd.Series(
                        [1, 1, 1, 1, 1, 1], dtype=RLEDtype(np.int64)
                    ),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
            # threshold
            0.0,
            # expected
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series([1, 2, 3, 4, 5, 6], dtype=np.int64),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame({"x": pd.Series([], dtype=np.int64)}),
            # threshold
            0.0,
            # expected
            pd.DataFrame({"x": pd.Series([], dtype=np.int64)}),
        ),
        (
            # orig
            pd.DataFrame({"x": pd.Series([], dtype=np.int64)}),
            # threshold
            0.1,
            # expected
            pd.DataFrame({"x": pd.Series([], dtype=RLEDtype(np.int64))}),
        ),
        (
            # orig
            pd.DataFrame(
                {
                    "single_value": pd.Series([1, 1, 1, 1, 1, 1], dtype=np.int64),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series(
                        [1, 2, 3, 4, 5, 6], dtype=RLEDtype(np.int64)
                    ),
                }
            ),
            # threshold
            0.5,
            # expected
            pd.DataFrame(
                {
                    "single_value": pd.Series(
                        [1, 1, 1, 1, 1, 1], dtype=RLEDtype(np.int64)
                    ),
                    "two_values": pd.Series([1, 1, 1, 2, 2, 2], dtype=np.int64),
                    "increasing": pd.Series(
                        [1, 2, 3, 4, 5, 6], dtype=RLEDtype(np.int64)
                    ),
                }
            ),
        ),
        (
            # orig
            pd.DataFrame({"x": pd.Series(range(10), dtype=np.int64)}),
            # threshold
            1.0,
            # expected
            pd.DataFrame({"x": pd.Series(range(10), dtype=np.int64)}),
        ),
        (
            # orig
            pd.DataFrame(),
            # threshold
            None,
            # expected
            pd.DataFrame(),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:.*would use a DatetimeBlock:UserWarning")
def test_auto_convert_to_rle_ok(
    orig: pd.DataFrame, threshold: Optional[float], expected: pd.DataFrame
) -> None:
    actual = auto_convert_to_rle(orig, threshold)
    pdt.assert_frame_equal(actual, expected)


def test_datetime_warns() -> None:
    df = pd.DataFrame(
        {
            "i1": pd.Series([1], dtype=np.int64),
            "d1": pd.Series([pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"),
            "i2": pd.Series([1], dtype=np.int64),
            "d2": pd.Series([pd.Timestamp("2020-01-01")], dtype="datetime64[ns]"),
        }
    )
    with pytest.warns(None) as record:
        auto_convert_to_rle(df, 0.5)
    assert len(record) == 2
    assert (
        str(record[0].message)
        == "Column d1 would use a DatetimeBlock and can currently not be RLE compressed."
    )
    assert (
        str(record[1].message)
        == "Column d2 would use a DatetimeBlock and can currently not be RLE compressed."
    )


def test_auto_convert_to_rle_threshold_out_of_range() -> None:
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match=r"threshold \(-0.1\) must be non-negative"):
        auto_convert_to_rle(df, -0.1)
