#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Refuse a non-x86_64 host
[ "$(uname -m)" = x86_64 ] || {
  echo "Error: fsim-lib runs on x86_64 only, not $(uname -m)"
  exit 1
}
IMAGE_NAME="fsim-lib:latest-amd64"

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
