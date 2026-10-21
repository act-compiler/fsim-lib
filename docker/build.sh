#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"
source ./fsim-env.env

# Detect architecture
case "$(uname -m)" in
  x86_64) ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  *)
    echo "Error: Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac
DOCKERFILE="Dockerfile"
IMAGE_NAME="fsim-lib:latest-${ARCH}"
BASE_OS="ubuntu:24.04"
# The env image to pull, and the one --env builds
ENV_IMAGE_NAME="devanshdvj/fsim-lib-env:${FSIM_ENV_TAG}-${ARCH}"
ENV_BUILD_IMAGE_NAME="devanshdvj/fsim-lib-env:${FSIM_ENV_BUILD_TAG}-${ARCH}"
# The CPU tarballs the env bakes in, from the library's dist/
DIST_DIR="../fsim/third_party/dist"

usage() {
  cat <<EOF
Usage: $0 [-e|--env] [-r|--rebuild] [-f|--force] [-y|--no-confirm] [-h|--help]

Builds ${IMAGE_NAME} on ${ENV_IMAGE_NAME},
pulled, or built here when the pull fails.

Options:
  -e, --env         Build only the env image ${ENV_BUILD_IMAGE_NAME}.
  -r, --rebuild     Remove old image (and optionally containers with --force) and rebuild.
  -f, --force       Remove containers referencing ${IMAGE_NAME} (only useful with --rebuild).
  -y, --no-confirm  Do not prompt for confirmation when --force is used.
  -h, --help        Show this help message.
EOF
}

# Parse args
ENV_ONLY=0
REBUILD=0
FORCE=0
NO_CONFIRM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --env)
      ENV_ONLY=1
      shift
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-confirm)
      NO_CONFIRM=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    -[!-]*)
      # support combined short flags e.g. -rf or -rfy
      flags="${1#-}"
      for ((i = 0; i < ${#flags}; i++)); do
        ch="${flags:i:1}"
        case "$ch" in
          e) ENV_ONLY=1 ;;
          r) REBUILD=1 ;;
          f) FORCE=1 ;;
          y) NO_CONFIRM=1 ;;
          h)
            usage
            exit 0
            ;;
          *)
            echo "Unknown flag: -${ch}"
            echo
            usage
            exit 1
            ;;
        esac
      done
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo
      usage
      exit 1
      ;;
  esac
done

# The env alone is the image to check and build
if [ "${ENV_ONLY}" -eq 1 ]; then
  IMAGE_NAME="${ENV_BUILD_IMAGE_NAME}"
fi

# The tarballs: every one setup.sh pins prebuilt for this arch
TARBALLS=()
for f in $(sed -n '/^prebuilt="$/,/^"$/s/^\([^ ]*\) [0-9a-f]\{64\}$/\1/p' ../fsim/setup.sh); do
  [[ "${f}" != *"-linux-$(uname -m).tar.gz" ]] || TARBALLS+=("${f}")
done
MISSING=()
for f in "${TARBALLS[@]}"; do
  [ -f "${DIST_DIR}/${f}" ] || MISSING+=("${f}")
done
# A release env holds every tarball
if [ "${ENV_ONLY}" -eq 1 ] && [ "${#MISSING[@]}" -gt 0 ]; then
  echo "Error: fsim/third_party/dist/ lacks ${MISSING[*]}; ./scripts/setup.sh with its --pjrt or --mlir builds each. Aborting."
  exit 1
fi

# Does an old image exist?
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Found existing image ${IMAGE_NAME}."
  IMAGE_EXISTS=1
else
  echo "No existing image named ${IMAGE_NAME} found."
  IMAGE_EXISTS=0
fi

# If image exists and user didn't ask to rebuild, just inform and exit
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 0 ]; then
  echo
  echo "Image ${IMAGE_NAME} already exists. To replace it, re-run with --rebuild. Aborting."
  exit 0
fi

# If image exists and user asked to rebuild, attempt to remove it
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 1 ]; then
  echo "--rebuild specified: will attempt to remove existing image ${IMAGE_NAME}."
  # List containers that reference this image (do not remove automatically unless --force)
  CONTAINERS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}	{{.Names}}	{{.Status}}') || true
  if [ -n "${CONTAINERS}" ]; then # Containers reference this image
    echo
    echo "Containers referencing image ${IMAGE_NAME}:"
    echo -e "CONTAINER_ID\tNAME\tSTATUS"
    echo "${CONTAINERS}"
    echo
    if [ "${FORCE}" -eq 1 ]; then
      if [ "${NO_CONFIRM}" -eq 0 ]; then
        read -rp "You passed --force: remove the above containers now? [y/N] " RESP
        case "${RESP}" in
          y | Y)
            echo "--force confirmed: removing containers that reference ${IMAGE_NAME}."
            ;;
          *)
            echo "--force not confirmed: Aborting."
            exit 1
            ;;
        esac
      else
        echo "--force and --no-confirm specified: removing containers that reference ${IMAGE_NAME}."
      fi
      # remove by container id
      docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}' | xargs -r docker rm -f
    else
      echo "Rebuild requested but containers reference this image. Re-run with --rebuild --force to remove them automatically, or remove them manually. Aborting."
      exit 1
    fi
  else # No containers reference this image
    echo
    echo "No containers reference ${IMAGE_NAME}."
  fi
  echo
  echo "Proceeding to remove image ${IMAGE_NAME}..."
  docker image rm "${IMAGE_NAME}"
fi

# The dist context: the tarballs and sidecars present, else empty
DIST_CTX="$(mktemp -d)"
trap 'rm -rf "${DIST_CTX}"' EXIT
for f in "${TARBALLS[@]}"; do
  for p in "${DIST_DIR}/${f}" "${DIST_DIR}/${f}.sha256"; do
    [ ! -f "${p}" ] || cp "${p}" "${DIST_CTX}/"
  done
done

# Build one stage of the Dockerfile as an image, with no provenance or SBOM attestation to push
build_stage() {
  echo
  echo "Building $2..."
  docker build -f "${DOCKERFILE}" --target "$1" -t "$2" --provenance=false --sbom=false --build-arg OS="${BASE_OS}" \
    --build-arg FSIM_ENV_IMAGE="${ENV_IMAGE_NAME}" --build-arg FSIM_ENV_TAG="${FSIM_ENV_BUILD_TAG}" \
    --build-context dist="${DIST_CTX}" .
}

# The env alone, or the local stage on it
if [ "${ENV_ONLY}" -eq 1 ]; then
  build_stage env "${IMAGE_NAME}"
else
  # The env: local, pulled, else built if it is the Dockerfile's own
  if ! docker image inspect "${ENV_IMAGE_NAME}" >/dev/null 2>&1 && ! docker pull "${ENV_IMAGE_NAME}"; then
    if [ "${FSIM_ENV_TAG}" != "${FSIM_ENV_BUILD_TAG}" ]; then
      echo "Error: cannot pull ${ENV_IMAGE_NAME}, and the Dockerfile defines ${FSIM_ENV_BUILD_TAG}. Aborting."
      exit 1
    fi
    echo "Cannot pull ${ENV_IMAGE_NAME}: building it here."
    [ "${#MISSING[@]}" -eq 0 ] || echo "Warning: fsim/third_party/dist/ lacks ${MISSING[*]}, so the env holds no copy of them."
    build_stage env "${ENV_IMAGE_NAME}"
  fi
  build_stage local "${IMAGE_NAME}"
fi

# Prune dangling images
echo "Pruning dangling images..."
docker image prune -f || true

echo
echo "Docker image ${IMAGE_NAME} built successfully."
