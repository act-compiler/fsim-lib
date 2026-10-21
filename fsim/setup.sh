#!/bin/bash
# Sets up one PJRT entry and the MLIR toolchain under fsim/third_party/.
# An entry is the PJRT C API header of its XLA commit, pinned by sha256, and its plugin.
# A -cpu plugin is unpacked from its dist/ tarball, else a prebuilt one, else a build
# from jax's Bazel workspace at the entry's tag; a build packs the tarball into dist/.
# A -cuda12 plugin and its CUDA come from PyPI wheels, pinned by sha256 and kept in dist/.
# The toolchain is MLIR and StableHLO's static libraries and headers at StableHLO's pins,
# unpacked the same way, with a build from source; a build packs the tarball into dist/.
# A prebuilt tarball is pinned by sha256 and copied into dist/ from /opt/fsim/dist.
#   ./fsim/setup.sh [--pjrt NAME] [--mlir]
set -euo pipefail
cd "$(dirname "$0")"

# The pins are x86_64 only.
[ "$(uname -m)" = x86_64 ] || {
  echo "setup.sh: x86_64 only, not $(uname -m)" >&2
  exit 1
}

usage() {
  echo "usage: $0 [--pjrt NAME] [--mlir]"
  echo "  NAME: jax-0.11.2-cpu (default), jax-0.4.33-cpu, jax-0.11.2-cuda12 or jax-0.4.33-cuda12"
  echo "  --pjrt fetches the entry's header, then unpacks its plugin from third_party/dist/:"
  echo "  a -cpu tarball, built there first when absent, or -cuda12 wheels, fetched there first"
  echo "  --mlir unpacks the toolchain's tarball from third_party/dist/, built there first when absent"
  echo "  an absent tarball first comes prebuilt from /opt/fsim/dist when pinned"
  echo "  no flag sets up the default entry and the toolchain"
}

name=jax-0.11.2-cpu
want_pjrt=
want_mlir=
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
      want_pjrt=1
      shift 2
      ;;
    --mlir)
      want_mlir=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done
# No flag sets up both.
[ -n "$want_pjrt$want_mlir" ] || want_pjrt=1 want_mlir=1

# The pins per jax release: tag commit, XLA commit, header, CPU build flags.
case "$name" in
  jax-0.11.2-cpu | jax-0.11.2-cuda12)
    jax=32544801e26115ac1794926d027148abf3baf009
    xla=91888df6ce85102c30220e41d952065925e10886
    header_sha=ddedb2e7b4df1d19a8d15c23e41298a3f41d48076707bd563d672f6a95736213
    flags=(--config=ci_linux_x86_64 --config=public_cache --define=xnn_enable_avxvnniint8=false)
    ;;
  jax-0.4.33-cpu | jax-0.4.33-cuda12)
    jax=80e1c94de63e7f89667cdf35f38d8fe298e97a50
    xla=720b2c53346660e95abbed7cf3309a8b85e979f9
    header_sha=70f0abfebb76464b2132eb1aa86508dd7e30bb7f7b4add6d4e2e5d8bbbc6deb9
    # The image's clang, as build.py --use_clang sets it, then the CPU flags.
    flags=(--action_env=CLANG_COMPILER_PATH=/usr/bin/clang-18 --repo_env=CC=/usr/bin/clang-18 --repo_env=CXX=/usr/bin/clang++-18
      --repo_env=BAZEL_COMPILER=/usr/bin/clang-18 --copt=-Wno-error=unused-command-line-argument --copt=-Wno-gnu-offsetof-extensions
      --config=avx_posix --config=mkl_open_source_only --strategy=Genrule=standalone)
    ;;
  *)
    echo "unknown PJRT entry: $name" >&2
    usage >&2
    exit 1
    ;;
esac
# The jax release, which names the tarball and the wheels.
release=${name%-*}

# The bazelisk pin.
bazelisk_sha=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992

# The layout: entry, dist tarball, build scratch.
out=third_party/pjrt/$name
header=$out/include/xla/pjrt/c/pjrt_c_api.h
case "$name" in
  *-cpu) so=pjrt_c_api_cpu_plugin.so ;;
  *) so=xla_cuda_plugin.so ;;
esac
pack=pjrt-cpu-$release-linux-x86_64
dist=third_party/dist
build=$PWD/third_party/pjrt/build/$name

# The GPU wheels, each a release or all, a file and its sha256:
# jax's plugin, then the CUDA of the with-cuda extra of jax-cuda12-plugin.
wheels="
jax-0.11.2 jax_cuda12_pjrt-0.11.2-py3-none-manylinux_2_27_x86_64.whl 2133b3afc63307c016ae4980f166b2b3de199759363ead74f7d550eaae953463
jax-0.4.33 jax_cuda12_pjrt-0.4.33-py3-none-manylinux2014_x86_64.whl b43f199ec27fd9b3bb79b34ed297894ad50bb7a6eab62012baaa9ea6607b22de
jax-0.11.2 nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 210cf05005a447e29214e9ce50851e83fc5f4358df8b453155d5e1918094dcb4
jax-0.11.2 nvidia_nvshmem_cu12-3.8.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl b2e5760d6158aefe7bb74348d5128bb1b46a7dd1fd4b7b1e6fe55b4b07888f70
all nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702
all nvidia_cuda_cupti_cu12-12.9.79-py3-none-manylinux_2_25_x86_64.whl 096bcf334f13e1984ba36685ad4c1d6347db214de03dbb6eebb237b41d9d934f
all nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 5d6a0d32fdc7ea39917c20065614ae93add6f577d840233237ff08e9a38f58f0
all nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3
all nvidia_cudnn_cu12-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl 0a4aa3a7d2264256506c6857fc41fc0c499982f78d70195b1cfcc9055c1957cd
all nvidia_cufft_cu12-11.4.1.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl c67884f2a7d276b4b80eb56a79322a95df592ae5e765cf1243693365ccab4e28
all nvidia_cusolver_cu12-11.7.5.82-py3-none-manylinux_2_27_x86_64.whl 15da72d1340d29b5b3cf3fd100e3cd53421dde36002eda6ed93811af63c40d88
all nvidia_cusparse_cu12-12.5.10.65-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 73060ce019ac064a057267c585bf1fd5a353734151f87472ff02b2c5c9984e78
all nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_x86_64.whl bb94b2348e4861d5e8aa00b36c3d12d9bdf9c6369b82cbb723de0b582280a21c
all nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl e3f1171dbdc83c5932a45f0f4c99180a70de9bd2718c1ab77d14104f6d7147f9
"
# This entry's wheels, a file and a sha256 per line.
pins=$(awk -v r="$release" '$1 == r || $1 == "all" { print $2, $3 }' <<<"$wheels")

# The toolchain pins: StableHLO's tag and commit, and its LLVM commit.
stablehlo_tag=v1.20.0
stablehlo=639335932274930617d9724fb25ec9f2f52c5e40
llvm=ab547095ead5464dc024d66264d9b8a987f429f3
# The toolchain's targets, built with their link closure.
mlir_targets=(MLIRParser MLIRBytecodeWriter MLIRBytecodeReader MLIRTransforms MLIRFuncDialect MLIRPass MLIRSupport
  StablehloOps StablehloPasses StablehloOptimizationPasses StablehloSerialization StablehloPortableApi VhloOps Version)

# The toolchain layout: install, dist tarball, build scratch.
mlir=third_party/mlir
mlir_pack=mlir-$stablehlo_tag-linux-x86_64
mlir_build=$PWD/$mlir/build

# The prebuilt CPU tarballs, each a file and its sha256.
prebuilt="
pjrt-cpu-jax-0.11.2-linux-x86_64.tar.gz e7f07bbdff2e67d1b0766c1a8f5632d39af7c0fcab72586de4928c25a5816692
pjrt-cpu-jax-0.4.33-linux-x86_64.tar.gz 873129acd6df14c07f1e2b2ab0f0e616a763f00a38a7bfc56b918563e503c1a4
mlir-v1.20.0-linux-x86_64.tar.gz e3e2be748e1e8f7560d6a85a916df05d9b268e724538b89731d013de82f724fa
"
# The env image's copy of them.
image_dist=/opt/fsim/dist

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
    fetch "$build/bin/bazelisk" "https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64" "$bazelisk_sha"
    chmod +x "$build/bin/bazelisk"
  }

  # The jax workspace at its tag's commit.
  fetch_commit jax-ml/jax "$build/jax" "$jax"

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

  # The tarball, then no build state.
  pack_dir "$build" "$pack"
  rm -rf "$build"
  rmdir --ignore-fail-on-non-empty "$(dirname "$build")"
}

# Fetch the entry's wheels into dist/, then unpack the plugin and CUDA.
unpack_wheels() {
  local file sha project
  mkdir -p "$dist"
  while read -r file sha; do
    # The wheel's PyPI project, from its file name.
    project=${file%%-*}
    project=${project//_/-}
    matches "$dist/$file" "$sha" || fetch "$dist/$file" "https://files.pythonhosted.org/packages/py3/${project:0:1}/$project/$file" "$sha"
  done <<<"$pins"

  # The unpack, skipped when lib/ came from these wheels.
  [ -f "$out/lib/$so" ] && cmp -s "$out/lib/wheels.txt" <(echo "$pins") && return
  rm -rf "$out/lib" "$out/nvidia" "$out/wheel"
  mkdir -p "$out/lib"
  while read -r file sha; do
    case "$file" in
      jax_cuda12_pjrt-*)
        python3 -I -m zipfile -e "$dist/$file" "$out/wheel"
        cp "$out/wheel/jax_plugins/xla_cuda12/$so" "$out/lib/"
        rm -rf "$out/wheel"
        ;;
      *) python3 -I -m zipfile -e "$dist/$file" "$out" ;;
    esac
  done <<<"$pins"
  rm -rf "$out"/*.dist-info
  # zipfile drops the mode bits, so ptxas would not run.
  chmod +x "$out"/nvidia/*/bin/*
  echo "$pins" >"$out/lib/wheels.txt"
}

# Move a dist/ file's .part in, after its sidecar.
seal() {
  echo "$2  $1" >"$dist/$1.sha256"
  mv "$dist/$1.part" "$dist/$1"
}

# Pack a build's directory into a dist/ tarball.
pack_dir() {
  mkdir -p "$dist"
  tar -czf "$dist/$2.tar.gz.part" -C "$1" "$2"
  seal "$2.tar.gz" "$(sha256sum <"$dist/$2.tar.gz.part" | cut -c1-64)"
}

# Put a tarball into dist/: kept, else the image's pinned copy, else a build.
provide() {
  local file=$1.tar.gz sha
  # A kept tarball holds to its sidecar.
  if [ -f "$dist/$file" ]; then
    (cd "$dist" && sha256sum -c --status "$file.sha256") || {
      echo "sha256 mismatch: $dist/$file" >&2
      exit 1
    }
    return
  fi
  # The image's copy, when pinned, holds to its pin.
  sha=$(awk -v f="$file" '$1 == f { print $2 }' <<<"$prebuilt")
  if [ -n "$sha" ] && [ -f "$image_dist/$file" ]; then
    matches "$image_dist/$file" "$sha" || {
      echo "sha256 mismatch: $image_dist/$file is not the pinned $sha" >&2
      exit 1
    }
    echo "prebuilt: copying $image_dist/$file"
    mkdir -p "$dist"
    cp "$image_dist/$file" "$dist/$file.part"
    seal "$file" "$sha"
    return
  fi
  "$2"
}

# Unpack a dist/ tarball into a dir, skipped when the dir came from it.
unpack() {
  local tarball=$dist/$1.tar.gz
  cmp -s "$tarball.sha256" "$2/$1.tar.gz.sha256" && return
  rm -rf "$2"
  mkdir -p "$2"
  tar -xzf "$tarball" --no-same-owner --strip-components="$3" -C "$2"
  cp "$tarball.sha256" "$2/"
}

# Set up the entry: its header, then its plugin.
setup_pjrt() {
  # The header, skipped when already pinned.
  if ! matches "$header" "$header_sha"; then
    mkdir -p "$(dirname "$header")"
    fetch "$header" "https://raw.githubusercontent.com/openxla/xla/$xla/xla/pjrt/c/pjrt_c_api.h" "$header_sha"
  fi

  # A -cuda12 plugin from its wheels, a -cpu one from its tarball.
  if [ "$so" = xla_cuda_plugin.so ]; then
    unpack_wheels
  else
    provide "$pack" build_plugin
    unpack "$pack" "$out/lib" 2
  fi
  echo "pjrt: $header $out/lib/$so"
}

# Fetch one commit of a GitHub repo, refetched unless at the pin.
fetch_commit() {
  local git=(git -c safe.directory="$2" -C "$2")
  [ "$("${git[@]}" rev-parse -q --verify HEAD 2>/dev/null)" = "$3" ] || {
    rm -rf "$2"
    git init -q "$2"
    "${git[@]}" fetch -q --depth 1 "https://github.com/$1.git" "$3"
    "${git[@]}" -c advice.detachedHead=false checkout -q FETCH_HEAD
  }
}

# Copy the headers of source dirs under a root, keeping their paths.
copy_headers() {
  local root=$1 to=$2
  shift 2
  (cd "$root" && find "$@" \( -name '*.h' -o -name '*.inc' -o -name '*.def' \) -exec cp --parents -t "$to" {} +)
}

# Build the toolchain from source, then pack it into dist/.
build_mlir() {
  local src=$mlir_build/src bin=$mlir_build/llvm
  local to=$mlir_build/$mlir_pack
  fetch_commit llvm/llvm-project "$src/llvm" "$llvm"
  fetch_commit openxla/stablehlo "$src/stablehlo" "$stablehlo"

  # LLVM and MLIR with StableHLO as an external project, no tools or tests.
  echo "mlir: building StableHLO $stablehlo_tag, about 10 min"
  cmake -G Ninja -S "$src/llvm/llvm" -B "$bin" -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_USE_LINKER=/usr/bin/ld.lld-18 \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_LIBXML2=OFF \
    -DMLIR_INCLUDE_TESTS=OFF -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_EXTERNAL_PROJECTS=stablehlo -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR="$src/stablehlo" \
    -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF -DMLIR_INCLUDE_DIRS="$src/llvm/mlir/include"
  ninja -C "$bin" -j"$(nproc)" "${mlir_targets[@]}"

  # The source headers, then the generated ones.
  rm -rf "$to"
  mkdir -p "$to/include" "$to/lib"
  copy_headers "$src/llvm/llvm/include" "$to/include" llvm llvm-c
  copy_headers "$src/llvm/mlir/include" "$to/include" mlir mlir-c
  copy_headers "$src/stablehlo" "$to/include" stablehlo
  copy_headers "$bin/include" "$to/include" llvm
  copy_headers "$bin/tools/mlir/include" "$to/include" mlir
  copy_headers "$bin/tools/stablehlo" "$to/include" stablehlo
  # The static libraries, less the build's own tablegen.
  find "$bin/lib" -maxdepth 1 -name '*.a' ! -name '*TableGen*' ! -name '*Tblgen*' -exec cp -t "$to/lib" {} +
  printf 'stablehlo %s %s\nllvm %s\n' "$stablehlo_tag" "$stablehlo" "$llvm" >"$to/pins.txt"

  # The tarball, then no build state.
  pack_dir "$mlir_build" "$mlir_pack"
  rm -rf "$mlir_build"
}

# Set up the toolchain from its tarball.
setup_mlir() {
  provide "$mlir_pack" build_mlir
  unpack "$mlir_pack" "$mlir" 1
  echo "mlir: $mlir/include $mlir/lib"
}

[ -z "$want_pjrt" ] || setup_pjrt
[ -z "$want_mlir" ] || setup_mlir
