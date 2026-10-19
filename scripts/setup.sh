#!/bin/bash
# Sets up a PJRT entry and the MLIR toolchain of the library in the container.
# Runs fsim/setup.sh, which skips what fsim/third_party/ already holds and builds
# a CPU plugin or the toolchain whose tarball is not in fsim/third_party/dist/.
#   ./scripts/setup.sh [--pjrt NAME] [--mlir]    no flag sets up both
#   NAME: jax-0.11.2-cpu (default), jax-0.4.33-cpu, jax-0.11.2-cuda12, jax-0.4.33-cuda12
set -euo pipefail
cd "$(dirname "$0")/.."

# The arguments pass through quoted; fsim/setup.sh parses them.
./docker/exec.sh ./fsim/setup.sh "${@@Q}"
