#!/bin/bash
# Fetches the library's third-party binaries into fsim/third_party/.
# The PJRT CPU plugin from zml/pjrt-artifacts and the PJRT C API header
# of the XLA commit it was built from, each pinned by sha256.
#   ./fsim/setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# The pins: XLA commit, zml release, per-arch plugin.
xla=6b73c4c49eb0c5aee1963c1e34c7cc8ba92ac202
header_sha=a38827698466b7ac81b2402ad144d027a8566714d94d649fba74a37be464ece4
release=https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-09-08T13-23-00Z
case "$(uname -m)" in
  x86_64)
    asset=pjrt-cpu_linux-amd64.tar.gz
    asset_sha=f99324db6146208672f5b1d420333d0ea26e3968d7abfa83a802b40276f855d3
    so_sha=edc5ecc1b13eceeba4b0c312a05efbfdd9bd0732040990024beacb64cc9c4569
    ;;
  aarch64)
    asset=pjrt-cpu_linux-arm64.tar.gz
    asset_sha=06884bda9f029999a58fb5f9cbe207c436c39ecaf7ef904600f6b3e92aba2926
    so_sha=a322e7983170881190d1a200e06ca4579945fe158587c4ad9f02f2be065c877f
    ;;
  *)
    echo "no pinned PJRT plugin for $(uname -m)" >&2
    exit 1
    ;;
esac
out=third_party/pjrt
header=$out/include/xla/pjrt/c/pjrt_c_api.h
so=$out/lib/libpjrt_cpu.so

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

# The header, skipped when already pinned.
if ! matches "$header" "$header_sha"; then
  mkdir -p "$(dirname "$header")"
  fetch "$header" "https://raw.githubusercontent.com/openxla/xla/$xla/xla/pjrt/c/pjrt_c_api.h" "$header_sha"
fi

# The plugin, extracted from its archive and checked again.
if ! matches "$so" "$so_sha"; then
  mkdir -p "$out/lib"
  fetch "$out/lib/$asset" "$release/$asset" "$asset_sha"
  tar -xzf "$out/lib/$asset" --no-same-owner -C "$out/lib" ./libpjrt_cpu.so
  rm "$out/lib/$asset"
  matches "$so" "$so_sha" || {
    echo "sha256 mismatch: $so" >&2
    exit 1
  }
fi
echo "pjrt: $header $so"
