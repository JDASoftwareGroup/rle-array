#!/usr/bin/env bash

set -exuo pipefail

mypy .
pytest
black --check .
isort --recursive --check-only
flake8
asv --config ./asv_bench/asv.conf.json run --show-stderr --environment existing --quick
python setup.py build_sphinx
