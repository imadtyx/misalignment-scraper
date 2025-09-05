#!/usr/bin/env bash
set -euo pipefail

PYBIN="${1:-python}"
"$PYBIN" src/mis-scrape/main.py --config src/mis-scrape/config.batch.yaml

