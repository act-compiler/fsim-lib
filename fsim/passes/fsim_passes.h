#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace fsim {

// Adds the library's own StableHLO passes to pm.
void addPasses(mlir::PassManager &pm);

// Sinks byte-grid bitcasts below slices, reshapes and updates.
std::unique_ptr<mlir::Pass> createByteGridsPass();
// Forwards slices over the updates of their chain.
std::unique_ptr<mlir::Pass> createForwardSlicesPass();
// Removes updates that later updates overwrite.
std::unique_ptr<mlir::Pass> createDeadUpdatesPass();

} // namespace fsim
