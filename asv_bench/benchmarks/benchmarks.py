import warnings
from contextlib import contextmanager

from pandas.errors import PerformanceWarning

from rle_array.testing import const_col, dim_col, generate_test_dataframe


class Base:
    min_run_count = 10
    processes = 1
    repeat = 5
    sample_time = 1.0
    warmup_time = 1.0

    def setup(self):
        self.df_baseline = generate_test_dataframe(n_dims=3, size=100)
        self.df_rle = self.df_baseline.astype("RLEDtype[int64]")

    @contextmanager
    def ignore_performance_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=PerformanceWarning)
            yield


class TimeAccess(Base):
    def setup(self):
        super().setup()

        self.shuffle_dim2_unstable = self.df_baseline.sort_values(
            dim_col(2), kind="quicksort"
        ).index.values
        self.shuffle_dim2_stable = self.df_baseline.sort_values(
            dim_col(2), kind="mergesort"
        ).index.values

    def time_take_unstable_const12_base(self):
        self.df_baseline[const_col([1, 2])].take(self.shuffle_dim2_unstable)

    def time_take_unstable_const12_rle(self):
        self.df_rle[const_col([1, 2])].take(self.shuffle_dim2_unstable)

    def time_take_stable_const12_base(self):
        self.df_baseline[const_col([1, 2])].take(self.shuffle_dim2_stable)

    def time_take_stable_const12_rle(self):
        self.df_rle[const_col([1, 2])].take(self.shuffle_dim2_stable)


class TimeGroupByReduce(Base):
    def setup(self):
        super().setup()

        df_rle_wo_dims = self.df_rle.copy()
        for d in range(3):
            df_rle_wo_dims[dim_col(d)] = self.df_baseline[dim_col(d)].copy()
        self.df_rle_wo_dims = df_rle_wo_dims

    def time_groupby2_sum_const12_baseline(self):
        self.df_baseline.groupby(dim_col(2))[const_col([1, 2])].sum()

    def time_groupby2_sum_const12_rle(self):
        with self.ignore_performance_warnings():
            self.df_rle_wo_dims.groupby(dim_col(2))[const_col([1, 2])].sum()


class TimeSeriesReduce(Base):
    def time_sum_const12_baseline(self):
        self.df_baseline[const_col([1, 2])].sum()

    def time_sum_const12_rle(self):
        self.df_rle[const_col([1, 2])].sum()

    def time_sum_const012_baseline(self):
        self.df_baseline[const_col([0, 1, 2])].sum()

    def time_sum_const012_rle(self):
        self.df_rle[const_col([0, 1, 2])].sum()


class TimeShift(Base):
    def time_shift_int_const12_base(self):
        self.df_baseline[const_col([1, 2])].shift(periods=1, fill_value=1)

    def time_shift_int_const12_rle(self):
        self.df_rle[const_col([1, 2])].shift(periods=1, fill_value=1)

    def time_shift_float_const12_base(self):
        self.df_baseline[const_col([1, 2])].shift(periods=1)

    def time_shift_float_const12_rle(self):
        self.df_rle[const_col([1, 2])].shift(periods=1)


class TimeOperator(Base):
    def time_add_const12_baseline(self):
        self.df_baseline[const_col([1, 2])] + self.df_baseline[const_col([1, 2])]

    def time_add_const12_rle(self):
        self.df_rle[const_col([1, 2])] + self.df_rle[const_col([1, 2])]

    def time_eq_const12_baseline(self):
        self.df_baseline[const_col([1, 2])] == self.df_baseline[const_col([1, 2])]

    def time_eq_const12_rle(self):
        self.df_rle[const_col([1, 2])] == self.df_rle[const_col([1, 2])]
