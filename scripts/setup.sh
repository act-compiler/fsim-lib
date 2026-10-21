#!/bin/bash
# Sets up the library's MLIR toolchain and PJRT plugins in the container.
# Runs fsim/setup.sh, which skips what fsim/third_party/ already holds;
# its --help lists the names it takes.
set -euo pipefail
cd "$(dirname "$0")/.."

# The arguments pass through quoted; fsim/setup.sh parses them.
./docker/exec.sh ./fsim/setup.sh "${@@Q}"
