=========
rle-array
=========

.. image:: https://github.com/JDASoftwareGroup/rle-array/workflows/CI/badge.svg?branch=master
    :target: https://github.com/JDASoftwareGroup/rle-array/actions?query=branch%3Amaster+workflow%3ACI
    :alt: Build Status
.. image:: https://codecov.io/gh/JDASoftwareGroup/rle-array/branch/master/graph/badge.svg?token=y2q96vlHqc
    :target: https://codecov.io/gh/JDASoftwareGroup/rle-array
    :alt: Coverage Status

`Extension Array`_ for `Pandas`_ that implements `Run-length Encoding`_.


.. contents:: Table of Contents


Quick Start
***********

Some basic setup first:

>>> import pandas as pd
>>> pd.set_option("display.max_rows", 40)
>>> pd.set_option("display.width", None)

We need some example data, so let's create some pseudo-weather data:

>>> from rle_array.testing import generate_example
>>> df = generate_example()
>>> df.head(10)
        date  month  year    city    country   avg_temp   rain   mood
0 2000-01-01      1  2000  city_0  country_0  12.400000  False     ok
1 2000-01-02      1  2000  city_0  country_0   4.000000  False     ok
2 2000-01-03      1  2000  city_0  country_0  17.200001  False  great
3 2000-01-04      1  2000  city_0  country_0   8.400000  False     ok
4 2000-01-05      1  2000  city_0  country_0   6.400000  False     ok
5 2000-01-06      1  2000  city_0  country_0  14.400000  False     ok
6 2000-01-07      1  2000  city_0  country_0  14.300000   True     ok
7 2000-01-08      1  2000  city_0  country_0   6.800000  False     ok
8 2000-01-09      1  2000  city_0  country_0  10.100000  False     ok
9 2000-01-10      1  2000  city_0  country_0  -1.200000  False     ok

Due to the large number of attributes for locations and the date, the data size is quite large:

>>> df.memory_usage()
Index            128
date        32000000
month        4000000
year         8000000
city        32000000
country     32000000
avg_temp    16000000
rain         4000000
mood        32000000
dtype: int64
>>> df.memory_usage().sum()
160000128

To compress the data, we can use ``rle-array``:

>>> import rle_array
>>> df_rle = df.astype({
...     "city": "RLEDtype[object]",
...     "country": "RLEDtype[object]",
...     "month": "RLEDtype[int8]",
...     "mood": "RLEDtype[object]",
...     "rain": "RLEDtype[bool]",
...     "year": "RLEDtype[int16]",
... })
>>> df_rle.memory_usage()
Index            128
date        32000000
month        1188000
year          120000
city           32000
country           64
avg_temp    16000000
rain         6489477
mood        17153296
dtype: int64
>>> df_rle.memory_usage().sum()
72982965

This works better the longer the runs are. In the above example, it does not work too well for ``"rain"``.


Development Plan
****************

The development of ``rle-array`` has the following priorities (in decreasing order):

1. **Correctness:** All results must be correct. The `Pandas`_-provided test suite must pass. Approximation are not
   allowed.
2. **Transparency:** The user can use :class:`~rle_array.RLEDtype` and :class:`~rle_array.RLEArray` like other `Pandas`_
   types. No special parameters or extra functions are required.
3. **Features:** Support all features that `Pandas`_ offers, even if it is slow (but inform the user using a
   :class:`pandas.errors.PerformanceWarning`).
4. **Simplicity:** Do not use `Python C Extensions`_ or `Cython`_ (`NumPy`_ and `Numba`_ are allowed).
5. **Memory Reduction:** Do not decompress the encoded data when not required, try to do as many calculations directly
   on the compressed representation.
6. **Performance:** It should be quick, for large data ideally faster than working on the uncompressed data. Use
   `Numba`_ to speed up code.


Implementation
**************

Imagine the following data array:

+-------+------+
| Index | Data |
+=======+======+
| 1     | "a"  |
+-------+------+
| 2     | "a"  |
+-------+------+
| 3     | "a"  |
+-------+------+
| 4     | "x"  |
+-------+------+
| 5     | "c"  |
+-------+------+
| 6     | "c"  |
+-------+------+
| 7     | "a"  |
+-------+------+
| 8     | "a"  |
+-------+------+

There some data points valid for multiple entries in a row:

+-------+------+
| Index | Data |
+=======+======+
| 1     | "a"  |
+-------+      +
| 2     |      |
+-------+      +
| 3     |      |
+-------+------+
| 4     | "x"  |
+-------+------+
| 5     | "c"  |
+-------+      +
| 6     |      |
+-------+------+
| 7     | "a"  |
+-------+      +
| 8     |      |
+-------+------+

These sections are also called *runs* and can be encoded by their value and their length:

+--------+-------+
| Length | Value |
+========+=======+
| 3      | "a"   |
+--------+-------+
| 1      | "x"   |
+--------+-------+
| 2      | "c"   |
+--------+-------+
| 2      | "a"   |
+--------+-------+

This representation is called `Run-length Encoding`_. To integrate this encoding better with `Pandas`_ and `NumPy`_ and
to support operations like slicing and random access (e.g. via :func:`pandas.api.extensions.ExtensionArray.take`), we
store the end position (the cum-sum of the length column) instead of the length:

+--------------+-------+
| End-position | Value |
+==============+=======+
| 3            | "a"   |
+--------------+-------+
| 4            | "x"   |
+--------------+-------+
| 6            | "c"   |
+--------------+-------+
| 8            | "a"   |
+--------------+-------+

The value array is an :class:`numpy.ndarray` with the same dtype as the original data and the end-positions are an
:class:`numpy.ndarray` with the dtype ``int64``.


License
*******

Licensed under:

- MIT License (``LICENSE.txt`` or https://opensource.org/licenses/MIT)


.. _Cython: https://cython.org/
.. _Extension Array: https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extensionarray
.. _Numba: https://numba.pydata.org/
.. _NumPy: https://numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _Python C Extensions: https://docs.python.org/3/extending/building.html
.. _Run-length Encoding: https://en.wikipedia.org/wiki/Run-length_encoding
