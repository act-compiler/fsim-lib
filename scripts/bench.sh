#!/bin/bash
# Generates the nki benchmark cases in the container.
# Each --bench is a case name, a directory or a name prefix under
# benchmarks/nki/, every case when none.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 gen [--bench PATH]..."
  echo "  gen    regenerates each case's NEFF, inputs and golden"
  echo "  --bench repeats, and resolves under benchmarks/nki/"
}

action=
benches=()
while [ "$#" -gt 0 ]; do
  # The valued options need a value.
  case "$1" in
    --bench)
      [ "$#" -ge 2 ] || {
        echo "$1 needs a value" >&2
        usage >&2
        exit 1
      }
      ;;
  esac
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    gen)
      [ -z "$action" ] || {
        usage >&2
        exit 1
      }
      action=$1
      shift
      ;;
    --bench)
      benches+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done
[ -n "$action" ] || {
  usage >&2
  exit 1
}

# The benches pass through quoted; suite.py resolves them.
cmd="benchmarks/nki/gen/regen.sh ${benches[*]@Q}"
# No root-owned pycache.
./docker/exec.sh "PYTHONDONTWRITEBYTECODE=1 $cmd"
