import warnings
from contextlib import contextmanager
from typing import Generator

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from rle_array.autoconversion import auto_convert_to_rle, decompress
from rle_array.testing import const_col, dim_col, generate_test_dataframe


class Base:
    min_run_count = 10
    processes = 1
    repeat = 5
    sample_time = 1.0
    warmup_time = 1.0

    def gen_baseline(self) -> pd.DataFrame:
        return generate_test_dataframe(n_dims=3, size=100)

    def setup(self) -> None:
        self.df_baseline = self.gen_baseline()
        self.df_rle = self.df_baseline.astype("RLEDtype[int64]")

    @contextmanager
    def ignore_performance_warnings(self) -> Generator[None, None, None]:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=PerformanceWarning)
            yield


class TimeAutoConversion(Base):
    def time_auto_convert_to_rle_compress_all(self) -> None:
        auto_convert_to_rle(self.df_baseline)

    def time_auto_convert_to_rle_no_compression_allowed(self) -> None:
        auto_convert_to_rle(self.df_baseline, 0.0)

    def time_auto_convert_to_rle_already_compressed(self) -> None:
        auto_convert_to_rle(self.df_rle)

    def time_decompress_compressed(self) -> None:
        decompress(self.df_rle)

    def time_decompress_noop(self) -> None:
        decompress(self.df_baseline)


class TimeCompression(Base):
    def time_decompress_array_astype(self) -> None:
        with self.ignore_performance_warnings():
            self.df_rle[const_col([1, 2])].array.astype(np.int64)

    def time_decompress_to_numpy(self) -> None:
        with self.ignore_performance_warnings():
            self.df_rle[const_col([1, 2])].to_numpy()


class TimeTake(Base):
    def setup(self) -> None:
        super().setup()

        self.shuffle_dim2_unstable = self.df_baseline.sort_values(
            dim_col(2), kind="quicksort"
        ).index.values
        self.shuffle_dim2_stable = self.df_baseline.sort_values(
            dim_col(2), kind="mergesort"
        ).index.values

    def time_unstable_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].take(self.shuffle_dim2_unstable)

    def time_unstable_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].take(self.shuffle_dim2_unstable)

    def time_stable_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].take(self.shuffle_dim2_stable)

    def time_stable_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].take(self.shuffle_dim2_stable)


class TimeGroupByReduce(Base):
    def setup(self) -> None:
        super().setup()

        df_rle_wo_dims = self.df_rle.copy()
        for d in range(3):
            df_rle_wo_dims[dim_col(d)] = self.df_baseline[dim_col(d)].copy()
        self.df_rle_wo_dims = df_rle_wo_dims

    def time_key2_opsum_const12_baseline(self) -> None:
        self.df_baseline.groupby(dim_col(2))[const_col([1, 2])].sum()

    def time_key2_opsum_const12_rle(self) -> None:
        with self.ignore_performance_warnings():
            self.df_rle_wo_dims.groupby(dim_col(2))[const_col([1, 2])].sum()


class TimeSeriesReduce(Base):
    def time_sum_const12_baseline(self) -> None:
        self.df_baseline[const_col([1, 2])].sum()

    def time_sum_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].sum()

    def time_sum_const012_baseline(self) -> None:
        self.df_baseline[const_col([0, 1, 2])].sum()

    def time_sum_const012_rle(self) -> None:
        self.df_rle[const_col([0, 1, 2])].sum()


class TimeShift(Base):
    def time_int_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].shift(periods=1, fill_value=1)

    def time_int_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].shift(periods=1, fill_value=1)

    def time_float_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].shift(periods=1)

    def time_float_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].shift(periods=1)


class TimeUnique(Base):
    def time_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].unique()

    def time_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])].unique()


class TimeOperator(Base):
    def time_add_const12_baseline(self) -> None:
        self.df_baseline[const_col([1, 2])] + self.df_baseline[const_col([1, 2])]

    def time_add_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])] + self.df_rle[const_col([1, 2])]

    def time_eq_const12_baseline(self) -> None:
        self.df_baseline[const_col([1, 2])] == self.df_baseline[const_col([1, 2])]

    def time_eq_const12_rle(self) -> None:
        self.df_rle[const_col([1, 2])] == self.df_rle[const_col([1, 2])]


class TimeGenerateTestDataFrame(Base):
    def time(self) -> None:
        self.gen_baseline()


class TimeFactorize(Base):
    def time_const12_base(self) -> None:
        self.df_baseline[const_col([1, 2])].factorize()

    def time_const12_rle(self) -> None:
        with self.ignore_performance_warnings():
            self.df_rle[const_col([1, 2])].factorize()
