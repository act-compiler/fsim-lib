#!/bin/bash
# Removes generated files, picked from git's ignored list, never a tracked
# or an untracked-but-not-ignored file; each kind is matched by exact name,
# but a case's logs, every *.log beside its .gitkeep.
# The list is read with the host's git; git clean deletes in the container,
# since the build output is root-owned.
set -euo pipefail
# A failing git fails the path list too.
shopt -s inherit_errexit
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 [-n] [--build] [--third-party] [--gen]"
  echo "  always removes each benchmark case's run.log, out/ and other logs"
  echo "  always removes every ignored __pycache__/ directory"
  echo "  --build        also every ignored build/ directory"
  echo "  --third-party  also fsim/third_party/, implies --build"
  echo "  --gen          also every ignored file under benchmarks/, implies --build"
  echo "  -n             prints what would go, removes nothing"
}

dry=0
build=0
third=0
gen=0
for arg in "$@"; do
  case "$arg" in
    -h | --help)
      usage
      exit 0
      ;;
    -n) dry=1 ;;
    --build) build=1 ;;
    --third-party) third=1 build=1 ;;
    --gen) gen=1 build=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Ignored paths under one pathspec, directories collapsed.
ignored() {
  git clean -ndX -- "$1" | sed 's/^Would remove //'
}

# Each kind's paths, by exact name; work/ is never listed.
paths=$(
  {
    if [ "$gen" -eq 1 ]; then
      ignored benchmarks
    else
      # A case's run.log and out/, and every log beside its .gitkeep.
      ignored benchmarks | while read -r p; do
        case "$p" in
          */run.log | */out/) echo "$p" ;;
          *.log) [ ! -f "$(dirname "$p")/.gitkeep" ] || echo "$p" ;;
        esac
      done
    fi
    ignored . | awk '/(^|\/)__pycache__\/$/ && !/^(work|fsim\/third_party)\//'
    if [ "$build" -eq 1 ]; then
      ignored . | awk '/(^|\/)build\/$/ && !/^work\//'
    fi
    if [ "$third" -eq 1 ]; then
      ignored fsim/third_party
    fi
  } | sort -u
)
if [ -z "$paths" ]; then
  echo 'nothing to clean'
  exit 0
fi

if [ "$dry" -eq 1 ]; then
  echo "$paths"
  exit 0
fi

# The paths pass through quoted; git clean removes only ignored ones.
mapfile -t list <<<"$paths"
./docker/exec.sh "GIT_LITERAL_PATHSPECS=1 git clean -ffdX -- ${list[*]@Q}"
