#!/bin/bash
# Builds one target in the container.
# nki builds drivers/nki/ into drivers/nki/build/<compiler>-<variant>/;
# fsim builds the library into fsim/build/.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 <target> [--compiler C] [--variant X]"
  echo "  targets: nki, fsim"
  echo "  --compiler: nki's xla-0.11-cpu, xla-0.4-cpu, custom-cpu, xla-0.11-gpu, xla-0.4-gpu or custom-gpu, default custom-cpu"
  echo "  --variant: nki's eager, taidl, trigger or overlap, default overlap"
}

target=
compiler=custom-cpu
variant=overlap
picked=0
while [ $# -gt 0 ]; do
  # The valued options need a known value.
  case "$1" in
    --compiler | --variant)
      [ $# -ge 2 ] || {
        echo "$1 needs a value" >&2
        usage >&2
        exit 1
      }
      case "$1=$2" in
        --compiler=xla-0.11-cpu | --compiler=xla-0.4-cpu | --compiler=custom-cpu | --compiler=xla-0.11-gpu | --compiler=xla-0.4-gpu | --compiler=custom-gpu | --variant=eager | --variant=taidl | --variant=trigger | --variant=overlap) ;;
        *)
          echo "unknown value for $1: $2" >&2
          usage >&2
          exit 1
          ;;
      esac
      ;;
  esac
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --compiler)
      compiler=$2
      picked=1
      shift 2
      ;;
    --variant)
      variant=$2
      picked=1
      shift 2
      ;;
    nki | fsim)
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
# Only nki takes a compiler and a variant.
if [ "$target" = fsim ] && [ "$picked" -eq 1 ]; then
  echo "fsim takes no --compiler or --variant" >&2
  exit 1
fi

# The target's source, build directory and own definitions.
case "$target" in
  nki)
    src=drivers/nki
    build=$src/build/$compiler-$variant
    defs="-DFSIM_COMPILER=$compiler -DFSIM_VARIANT=$variant"
    ;;
  fsim)
    src=fsim
    build=$src/build
    defs=
    ;;
esac

# Configure and build the target.
./docker/exec.sh "cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release $defs -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld-18 -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld-18 && cmake --build $build"
