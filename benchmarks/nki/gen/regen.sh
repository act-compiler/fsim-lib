#!/bin/bash
# Regenerates nki cases: each source's gen.py on its own cases, in the NKI env.
# Benches resolve as suite.py resolves them; runs in the container.
#   benchmarks/nki/gen/regen.sh [bench...]    every case when none is named
set -euo pipefail
cd "$(dirname "$0")/../../.."

nki=/opt/conda/envs/nki/bin
start=$(date +%s)

# Resolve the benches; this also checks the lists against the folders.
resolved=$(python3 benchmarks/nki/suite.py "$@")
mapfile -t names <<<"$resolved"

# Each source's gen.py on the requested cases it lists.
for gen in benchmarks/nki/gen/*/gen.py; do
  source=$(basename "$(dirname "$gen")")
  todo=()
  for name in "${names[@]}"; do
    if [ "${name%%/*}" = "$source" ]; then
      todo+=("$name")
    fi
  done
  [ "${#todo[@]}" -gt 0 ] || continue
  echo "== $source: ${#todo[@]} cases"
  PATH="$nki:$PATH" "$nki/python" "$gen" "${todo[@]}"
done

echo "== regenerated ${#names[@]} cases in $(($(date +%s) - start)) s"
