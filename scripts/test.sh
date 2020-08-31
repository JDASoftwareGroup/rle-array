#!/usr/bin/env bash

set -exuo pipefail

mypy .
pytest
black --check .
isort --check-only .
flake8
asv run --show-stderr --environment existing --quick
python setup.py build_sphinx
