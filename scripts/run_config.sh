#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/config.yaml> [python_bin]" >&2
  exit 2
fi

CFG="$1"
PYBIN="${2:-python}"

"$PYBIN" src/mis-scrape/main.py --config "$CFG"

