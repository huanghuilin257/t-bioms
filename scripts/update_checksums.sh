#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$root/SHA256SUMS.txt"

find "$root" -type f \
  ! -path "$out" \
  ! -path '*/.git/*' \
  ! -path '*/__pycache__/*' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#  $root/#  #" > "$out"

echo "Wrote $out"
