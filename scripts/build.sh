#!/bin/bash
# Builds one target in the container.
# fsim builds the library into fsim/build/.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 <target>"
  echo "  targets: fsim"
}

target=
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    fsim)
      [ -z "$target" ] || {
        usage >&2
        exit 1
      }
      target=$1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done
[ -n "$target" ] || {
  usage >&2
  exit 1
}

# The target's source and build directory.
case "$target" in
  fsim)
    src=fsim
    build=$src/build
    ;;
esac

# Configure and build the target.
./docker/exec.sh "cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld-18 -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld-18 && cmake --build $build"
