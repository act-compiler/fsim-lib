// Removes the updates later ones overwrite: from each returned value up
// its chain of single-use updates, an update whose window the later
// windows cover is dead and reads as its operand.
#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "window.h"

namespace fsim {
namespace {

using mlir::stablehlo::DynamicUpdateSliceOp;

// The update a value's single-use views come from, or null.
DynamicUpdateSliceOp above(mlir::Value value) {
  while (value.hasOneUse()) {
    mlir::Operation *op = value.getDefiningOp();
    if (!op)
      return {};
    if (auto update = mlir::dyn_cast<DynamicUpdateSliceOp>(op))
      return update;
    if (!isView(op))
      return {};
    value = op->getOperand(0);
  }
  return {};
}

// Prunes the chain ending in value, tail to head.
void prune(mlir::Value value) {
  // Through views to the tail update.
  while (mlir::Operation *op = value.getDefiningOp()) {
    if (!isView(op))
      break;
    value = op->getOperand(0);
  }
  auto update = value.getDefiningOp<DynamicUpdateSliceOp>();
  Intervals later;
  while (update) {
    std::optional<Window> window = Window::of(update);
    const Intervals *bytes = window ? window->bytes() : nullptr;
    if (!bytes)
      return;
    mlir::Value operand = update.getOperand();
    if (covered(*bytes, later)) {
      update.getResult().replaceAllUsesWith(operand);
      update.erase();
    } else {
      unite(later, *bytes);
    }
    update = above(operand);
  }
}

struct DeadUpdates : mlir::PassWrapper<DeadUpdates, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeadUpdates)
  llvm::StringRef getArgument() const override {
    return "fsim-dead-updates";
  }
  void runOnOperation() override {
    llvm::SmallVector<mlir::func::ReturnOp> returns;
    getOperation().walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
    for (mlir::func::ReturnOp ret : returns)
      for (mlir::Value value : ret.getOperands())
        prune(value);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDeadUpdatesPass() {
  return std::make_unique<DeadUpdates>();
}

} // namespace fsim
