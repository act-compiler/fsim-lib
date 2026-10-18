#!/bin/bash
# Sets up one PJRT entry under fsim/third_party/: the PJRT C API header
# of its XLA commit, pinned by sha256, and the CPU plugin, unpacked from
# dist/ or else built from jax's Bazel workspace at the entry's tag.
# A build packs the tarball into dist/ and leaves no other state.
#   ./fsim/setup.sh [--pjrt NAME]
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 [--pjrt NAME]"
  echo "  NAME: jax-0.11.2 (default) or jax-0.4.33"
  echo "  fetches the entry's header, then unpacks its plugin from third_party/dist/, built there first when absent"
}

name=jax-0.11.2
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --pjrt)
      [ $# -gt 1 ] || {
        usage >&2
        exit 1
      }
      name=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# The pins per entry: jax tag and commit, XLA commit, header.
arch=$(uname -m)
case "$name" in
  jax-0.11.2)
    jax=32544801e26115ac1794926d027148abf3baf009
    xla=91888df6ce85102c30220e41d952065925e10886
    header_sha=ddedb2e7b4df1d19a8d15c23e41298a3f41d48076707bd563d672f6a95736213
    case "$arch" in
      x86_64) flags=(--config=ci_linux_x86_64 --config=public_cache --define=xnn_enable_avxvnniint8=false) ;;
      aarch64) flags=(--config=ci_linux_aarch64 --config=public_cache) ;;
    esac
    ;;
  jax-0.4.33)
    jax=80e1c94de63e7f89667cdf35f38d8fe298e97a50
    xla=720b2c53346660e95abbed7cf3309a8b85e979f9
    header_sha=70f0abfebb76464b2132eb1aa86508dd7e30bb7f7b4add6d4e2e5d8bbbc6deb9
    # The image's clang, as build.py --use_clang sets it.
    clang=(--action_env=CLANG_COMPILER_PATH=/usr/bin/clang-18 --repo_env=CC=/usr/bin/clang-18 --repo_env=CXX=/usr/bin/clang++-18
      --repo_env=BAZEL_COMPILER=/usr/bin/clang-18 --copt=-Wno-error=unused-command-line-argument --copt=-Wno-gnu-offsetof-extensions)
    case "$arch" in
      x86_64) flags=("${clang[@]}" --config=avx_posix --config=mkl_open_source_only --strategy=Genrule=standalone) ;;
      aarch64) flags=("${clang[@]}" --strategy=Genrule=standalone) ;;
    esac
    ;;
  *)
    echo "unknown PJRT entry: $name" >&2
    usage >&2
    exit 1
    ;;
esac
tag=jax-v${name#jax-}

# The bazelisk pin per arch.
case "$arch" in
  x86_64)
    bazelisk=bazelisk-linux-amd64
    bazelisk_sha=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
    ;;
  aarch64)
    bazelisk=bazelisk-linux-arm64
    bazelisk_sha=e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010
    ;;
  *)
    echo "no pinned PJRT build for $arch" >&2
    exit 1
    ;;
esac

# The layout: entry, dist tarball, build scratch.
out=third_party/pjrt/$name
header=$out/include/xla/pjrt/c/pjrt_c_api.h
so=pjrt_c_api_cpu_plugin.so
pack=pjrt-cpu-$name-linux-$arch
dist=third_party/dist
tarball=$dist/$pack.tar.gz
build=$PWD/third_party/pjrt/build/$name

# True when the file exists with this sha256.
matches() { [ -f "$1" ] && echo "$2  $1" | sha256sum -c --status; }

# Download a url to a file, failing on a sha256 mismatch.
fetch() {
  curl -fsSL -o "$1.part" "$2"
  matches "$1.part" "$3" || {
    echo "sha256 mismatch: $2" >&2
    exit 1
  }
  mv "$1.part" "$1"
}

# Build the plugin from source, then pack it into dist/.
build_plugin() {
  local bz=("$build/bin/bazelisk" --output_user_root="$build/root")
  local all=("${flags[@]}" --repo_env=HERMETIC_PYTHON_VERSION=3.12
    --repository_cache="$build/cache/repos" --jobs=$(($(nproc) * 3 / 4))
    --color=no --curses=no --show_progress_rate_limit=60
    # Keep the GetPjrtApi object, and export it.
    --linkopt=-Wl,--undefined=GetPjrtApi '--per_file_copt=.*xla/pjrt/c/pjrt_c_api_cpu\.cc@-fvisibility=default')
  local label=@xla//xla/pjrt/c:$so
  export HOME=$build/home XDG_CACHE_HOME=$build/cache BAZELISK_HOME=$build/cache/bazelisk
  mkdir -p "$build/bin" "$HOME"
  [ -x "$build/bin/bazelisk" ] || {
    fetch "$build/bin/bazelisk" "https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/$bazelisk" "$bazelisk_sha"
    chmod +x "$build/bin/bazelisk"
  }

  # The clone, checked against the tag's pinned commit.
  [ -d "$build/jax" ] || git -c advice.detachedHead=false clone -q --depth 1 --branch "$tag" https://github.com/jax-ml/jax.git "$build/jax"
  [ "$(git -c safe.directory="$build/jax" -C "$build/jax" rev-parse HEAD)" = "$jax" ] || {
    echo "$tag is not at $jax" >&2
    exit 1
  }

  # Bazel refuses an output root another uid owns.
  [ ! -d "$build/root" ] || chown -R "$(id -u):$(id -g)" "$build/root"
  echo "pjrt: building $name, 1-2 h"
  (
    cd "$build/jax"
    trap '"${bz[@]}" shutdown' EXIT
    "${bz[@]}" build "${all[@]}" "$label"
    mkdir -p "$build/$pack/lib"
    cp "$("${bz[@]}" cquery "${all[@]}" --output=files "$label")" "$build/$pack/lib/"
  )
  # A plugin without GetPjrtApi never packs.
  nm -D --defined-only --without-symbol-versions "$build/$pack/lib/$so" | grep ' T GetPjrtApi$' >/dev/null || {
    echo "pjrt: $so exports no GetPjrtApi" >&2
    exit 1
  }

  # The tarball and its sidecar, then no build state.
  mkdir -p "$dist"
  tar -czf "$tarball.part" -C "$build" "$pack"
  mv "$tarball.part" "$tarball"
  (cd "$dist" && sha256sum "$pack.tar.gz" >"$pack.tar.gz.sha256")
  rm -rf "$build"
  rmdir --ignore-fail-on-non-empty "$(dirname "$build")"
}

# The header, skipped when already pinned.
if ! matches "$header" "$header_sha"; then
  mkdir -p "$(dirname "$header")"
  fetch "$header" "https://raw.githubusercontent.com/openxla/xla/$xla/xla/pjrt/c/pjrt_c_api.h" "$header_sha"
fi

# The plugin, built only when dist/ lacks its tarball.
[ -f "$tarball" ] || build_plugin
(cd "$dist" && sha256sum -c --status "$pack.tar.gz.sha256") || {
  echo "sha256 mismatch: $tarball" >&2
  exit 1
}

# The unpack, skipped when lib/ came from this tarball.
if ! { [ -f "$out/lib/$so" ] && cmp -s "$tarball.sha256" "$out/lib/$pack.tar.gz.sha256"; }; then
  rm -rf "$out/lib"
  mkdir -p "$out/lib"
  tar -xzf "$tarball" --no-same-owner --strip-components=2 -C "$out/lib" "$pack/lib/$so"
  cp "$tarball.sha256" "$out/lib/"
fi
echo "pjrt: $header $out/lib/$so"
