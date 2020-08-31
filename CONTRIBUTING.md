# How To Contribute


## Bugs

If you've found a bug, please report it to the issue tracker and

* Describe the bug you encountered and what the expected behavior should be.
* Provide a [Minimal, Reproducible Example](https://stackoverflow.com/help/mcve) (if possible).
* Be as explicit about your environment as possible, e.g. provide a `pip freeze` / `conda list`.

## Code Contributions

**Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in the work, shall be
dual-licensed under MIT license, without any additional terms or conditions.**

Please file a GitHub pull request with your contribution. See the [Development](#Development) section for details on
tooling. See the "Development Plan" in the README for the generic prioritization.


## Development

### Installation
To get started, set up a new virtual environment and install all requirements:

```bash
virtualenv --python=python3.6 .venv
source .venv/bin/activate
pip install poetry
poetry install
```

### Code style

To ensure a consistent code style across the code base we're using the following tools:

- [`black`](https://github.com/psf/black): code formatter
- [`flake8`](https://gitlab.com/pycqa/flake8): linting
- [`isort`](https://github.com/timothycrosley/isort): sorting of imports

We have a convenience script that runs all these tools and a code style check for you:

```bash
poetry run ./scripts/fmt.sh
```

### Testing
There are different tools that ensure a well tested and presented library. To run them all at once (useful for
development), use:

```bash
poetry run ./scripts/test.sh
```

### Pytest
We're using [pytest](https://pytest.org) as a testing framework and make heavy use of `fixtures` and `parametrization`.
To run the tests simply run:

```bash
poetry run pytest
```

### Benchmarks
For performance critical code paths we have [asv](https://asv.readthedocs.io/) benchmarks in place in the subfolder
`benchmarks`. To run the benchmarks a single time and receive immediate feedback run:

```bash
poetry run asv run --python=same --show-stderr
```

### Documentation
Documentation is created using [Sphinx](https://www.sphinx-doc.org/) and can be build by using:

```bash
poetry run python setup.py build_sphinx
```

### Typing
We use [mypy](http://mypy-lang.org/) to check python types. It can be run using:

```bash
poetry run mypy .
```

## Performance Improvements
If you wish to contribute a performance improvement, please ensure that a benchmark (in `asv_bench`) exists or that you
provide on in your pull request. Please run that benchmark before and after your change and add both values to the
commit message of your contribution.
