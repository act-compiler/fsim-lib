#!/bin/bash
# Generates or runs one ISA's benchmark cases in the container.
# Each --bench is a case name, a directory or a name prefix under
# benchmarks/<isa>/, every case when none; the only ISA is nki.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 gen --isa <isa> [--bench PATH]..."
  echo "       $0 run --isa <isa> [--compiler C] [--variant V] [--trigger N] [--overlap T] [--bench PATH]..."
  echo "  gen    regenerates each case's kernel.trace and inputs, and its NEFF and golden"
  echo "  run    simulates each case's kernel.trace on one build and checks its outputs"
  echo "  --compiler  xla-0.11-cpu, xla-0.4-cpu, custom-cpu, xla-0.11-gpu, xla-0.4-gpu or custom-gpu, custom-cpu by default"
  echo "  --variant   eager, taidl, trigger or overlap, overlap by default"
  echo "  --trigger   op count that flushes, 10000 by default; trigger and overlap only"
  echo "  --overlap   compile threads, 4 by default; overlap only"
  echo "  isas: nki; --bench repeats, and resolves under benchmarks/<isa>/"
  echo "  FSIM_WALL_TIMEOUT caps each run's wall seconds, 60 by default"
}

action=
isa=
benches=()
knobs=()
while [ "$#" -gt 0 ]; do
  # The valued options need a value.
  case "$1" in
    --isa | --bench | --compiler | --variant | --trigger | --overlap)
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
    gen | run)
      [ -z "$action" ] || {
        usage >&2
        exit 1
      }
      action=$1
      shift
      ;;
    --isa)
      isa=$2
      shift 2
      ;;
    --bench)
      benches+=("$2")
      shift 2
      ;;
    --compiler | --variant | --trigger | --overlap)
      knobs+=("$1" "$2")
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
# Only run takes the build and its knobs.
if [ "${#knobs[@]}" -gt 0 ] && [ "$action" != run ]; then
  echo "$action takes no --compiler, --variant, --trigger or --overlap" >&2
  exit 1
fi
case "$isa" in
  nki) ;;
  '')
    echo "--isa is required" >&2
    exit 1
    ;;
  *)
    echo "unknown isa: $isa" >&2
    exit 1
    ;;
esac

# The benches and knobs pass through quoted; the tools resolve them.
case "$action" in
  gen) cmd="benchmarks/$isa/gen/regen.sh ${benches[*]@Q}" ;;
  run) cmd="/opt/conda/envs/nki/bin/python benchmarks/$isa/run.py ${knobs[*]@Q} ${benches[*]@Q}" ;;
esac
# No root-owned pycache; the wall cap only when set.
./docker/exec.sh "PYTHONDONTWRITEBYTECODE=1 ${FSIM_WALL_TIMEOUT:+FSIM_WALL_TIMEOUT=$FSIM_WALL_TIMEOUT} $cmd"
