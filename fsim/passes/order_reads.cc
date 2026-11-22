// Orders each read of a buffer before the update that overwrites it in
// place: the buffer threads through one optimization_barrier per read,
// beside its consumer, so XLA updates in place without a copy, and the
// read stays fused into its consumer.
#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "window.h"
#include "llvm/ADT/SetVector.h"

namespace fsim {
namespace {

using mlir::stablehlo::DynamicUpdateSliceOp;
using mlir::stablehlo::OptimizationBarrierOp;

// The reads of root past its views, the update aside; false on a fork.
bool reads(mlir::Value root, DynamicUpdateSliceOp update,
           llvm::SetVector<mlir::Operation *> &leaves) {
  llvm::SmallVector<mlir::Value> views{root};
  while (!views.empty()) {
    mlir::Value value = views.pop_back_val();
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == update.getOperation() && use.getOperandNumber() == 0)
        continue;
      // A use in another block, another update or a return bails out.
      if (user->getBlock() != update->getBlock() || mlir::isa<mlir::func::ReturnOp>(user) ||
          (mlir::isa<DynamicUpdateSliceOp>(user) && use.getOperandNumber() == 0))
        return false;
      if (isView(user))
        views.push_back(user->getResult(0));
      else
        leaves.insert(user);
    }
  }
  return true;
}

// Adds op and what it needs after update to moved, producers first;
// false when it needs the update, a region or a side effect.
bool hoistable(mlir::Operation *op, DynamicUpdateSliceOp update,
               llvm::SetVector<mlir::Operation *> &moved) {
  // Depth first, an op taken once its producers are.
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>> stack{{op, 0}};
  while (!stack.empty()) {
    auto &[top, next] = stack.back();
    if (top == update.getOperation())
      return false;
    if (!update->isBeforeInBlock(top) || moved.contains(top)) {
      stack.pop_back();
      continue;
    }
    if (top->getNumRegions() != 0 || !mlir::isMemoryEffectFree(top))
      return false;
    if (next == top->getNumOperands()) {
      moved.insert(top);
      stack.pop_back();
      continue;
    }
    mlir::Operation *producer = top->getOperand(next++).getDefiningOp();
    if (producer)
      stack.push_back({producer, 0});
  }
  return true;
}

// The op a read's single uses lead to, through the first update.
mlir::Operation *consumer(mlir::Operation *read, DynamicUpdateSliceOp update) {
  mlir::Operation *op = read;
  while (op->getNumResults() == 1 && op->getResult(0).hasOneUse() &&
         !mlir::isa<DynamicUpdateSliceOp>(op)) {
    mlir::Operation *user = *op->getResult(0).getUsers().begin();
    if (user == update.getOperation() || user->getBlock() != op->getBlock() ||
        user->hasTrait<mlir::OpTrait::IsTerminator>() || mlir::isa<OptimizationBarrierOp>(user))
      break;
    op = user;
  }
  return op;
}

// Threads the reads of update's buffer before it.
void order(DynamicUpdateSliceOp update) {
  // The buffer under the update's views.
  llvm::SmallVector<mlir::Operation *> path;
  mlir::Value root = update.getOperand();
  while (mlir::Operation *op = root.getDefiningOp()) {
    if (!isView(op))
      break;
    path.push_back(op);
    root = op->getOperand(0);
  }
  llvm::SetVector<mlir::Operation *> leaves;
  if (!reads(root, update, leaves) || leaves.empty())
    return;
  // Each read becomes its consumer where that moves before the update;
  // a read after the update moves before it, else the buffer stays as is.
  llvm::SetVector<mlir::Operation *> moved, ends;
  for (mlir::Operation *leaf : leaves) {
    mlir::Operation *end = consumer(leaf, update);
    size_t mark = moved.size();
    if (end != leaf && hoistable(end, update, moved)) {
      ends.insert(end);
      continue;
    }
    // A failed consumer hoist pops what it added.
    while (moved.size() > mark)
      moved.pop_back();
    if (!hoistable(leaf, update, moved))
      return;
    ends.insert(leaf);
  }
  for (mlir::Operation *op : moved)
    op->moveBefore(update);
  auto sorted = ends.takeVector();
  llvm::sort(sorted, [](mlir::Operation *a, mlir::Operation *b) { return a->isBeforeInBlock(b); });

  // Each read's results pass a barrier beside the buffer.
  mlir::OpBuilder builder(update);
  mlir::Value buffer = root;
  for (mlir::Operation *leaf : sorted) {
    builder.setInsertionPointAfter(leaf);
    llvm::SmallVector<mlir::Value> operands{buffer};
    operands.append(leaf->result_begin(), leaf->result_end());
    auto barrier = OptimizationBarrierOp::create(builder, leaf->getLoc(), operands);
    for (unsigned i = 0; i < leaf->getNumResults(); ++i)
      leaf->getResult(i).replaceAllUsesExcept(barrier->getResult(i + 1), barrier);
    buffer = barrier->getResult(0);
  }

  // The update's views, rebuilt over the threaded buffer.
  builder.setInsertionPoint(update);
  for (mlir::Operation *view : llvm::reverse(path)) {
    mlir::IRMapping map;
    map.map(view->getOperand(0), buffer);
    buffer = builder.clone(*view, map)->getResult(0);
  }
  update.setOperand(0, buffer);
}

struct OrderReads : mlir::PassWrapper<OrderReads, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrderReads)
  llvm::StringRef getArgument() const override {
    return "fsim-order-reads";
  }
  void runOnOperation() override {
    // A declaration has no block.
    if (getOperation().isExternal())
      return;
    llvm::SmallVector<DynamicUpdateSliceOp> updates;
    // The entry block's updates only.
    for (DynamicUpdateSliceOp update : getOperation().front().getOps<DynamicUpdateSliceOp>())
      updates.push_back(update);
    for (DynamicUpdateSliceOp update : updates)
      order(update);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createOrderReadsPass() {
  return std::make_unique<OrderReads>();
}

} // namespace fsim
