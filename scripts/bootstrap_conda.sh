#!/usr/bin/env bash

set -exuo pipefail

readonly CONDA_SH="$HOME/miniconda/etc/profile.d/conda.sh"

# bootstrap conda
if [ -e "$CONDA_SH" ]; then
    echo "conda exists, skip init."
else
    echo "conda not found, init..."
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
fi

# activate conda
# shellcheck disable=SC1090
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r

# configure conda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --set always_yes yes --set changeps1 no
conda config --set pip_interop_enabled True

# update conda itself
conda update -q conda

# Useful for debugging any issues with conda
conda info -a
