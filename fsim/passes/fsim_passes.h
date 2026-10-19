#pragma once

#include "mlir/Pass/PassManager.h"

namespace fsim {

// Adds the library's own StableHLO passes to pm.
void addPasses(mlir::PassManager &pm);

} // namespace fsim
