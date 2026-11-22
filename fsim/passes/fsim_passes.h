#pragma once

#include <cstdint>
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
// Orders reads of a buffer before its in-place update.
std::unique_ptr<mlir::Pass> createOrderReadsPass();
// Vmaps @main at batch_size, refusing an op without a rule.
std::unique_ptr<mlir::Pass> createVmapPass(int64_t batch_size);

} // namespace fsim
