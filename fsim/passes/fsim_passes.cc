#include "fsim_passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

namespace fsim {

void addPasses(mlir::PassManager &pm) {
  // Byte tensors as byte grids, then forward reads and drop dead writes.
  pm.addNestedPass<mlir::func::FuncOp>(createByteGridsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(createForwardSlicesPass());
  pm.addNestedPass<mlir::func::FuncOp>(createDeadUpdatesPass());
  // Forwarded tiles cancel their bitcasts; dead ops go.
  pm.addNestedPass<mlir::func::FuncOp>(createByteGridsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

} // namespace fsim
