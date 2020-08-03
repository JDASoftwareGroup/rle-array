#!/usr/bin/env bash

set -exuo pipefail

black .
isort --atomic .
