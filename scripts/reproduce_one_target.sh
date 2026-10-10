#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TARGET" >&2
  exit 2
fi

target=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}

if [[ ! -f "$root/data/$target/dataset.csv" ]]; then
  echo "Missing frozen dataset: $root/data/$target/dataset.csv" >&2
  exit 1
fi

export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"

# p01 is intentionally skipped: the release package starts from the frozen
# curated table and does not redistribute raw ChEMBL exports.
"$python_bin" pipelines/p02_features.py --target "$target"
"$python_bin" pipelines/p03_protocol_matrix.py --target "$target"
"$python_bin" pipelines/p04_bootstrap_ci.py --target "$target"
"$python_bin" pipelines/p05_ad_analysis.py --target "$target"
"$python_bin" pipelines/p06_report.py --target "$target"

