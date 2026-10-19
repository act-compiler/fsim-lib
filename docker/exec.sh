#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -m)" in
  x86_64) ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  *)
    echo "Error: Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac
IMAGE_NAME="fsim-lib:latest-${ARCH}"

HOST_MOUNT="$(pwd)/.."

GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 &&
  docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  GPU_FLAGS=(--gpus all)
fi

trap './chown.sh >/dev/null 2>&1' EXIT

docker run --rm --entrypoint bash \
  --name "fsim-lib-exec-$$-${RANDOM}" \
  "${GPU_FLAGS[@]}" \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -lc "$*"
