// Forwards each slice over the updates of its chain: past an update it
// misses, to that update's operand; onto an update that holds its window,
// as a slice or a reshape of the update's tile; past every update, onto
// the chain's head.
#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "window.h"
#include "llvm/ADT/DenseMap.h"

namespace fsim {
namespace {

using mlir::stablehlo::BitcastConvertOp;
using mlir::stablehlo::DynamicUpdateSliceOp;
using mlir::stablehlo::ReshapeOp;
using mlir::stablehlo::SliceOp;

// The most links one slice walks.
constexpr int kHops = 2048;

// The value under a value's reshapes.
mlir::Value unreshape(mlir::Value value) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>())
    value = reshape.getOperand();
  return value;
}

// Erases the views under value left without uses.
void drop(mlir::Value value) {
  while (mlir::Operation *op = value.getDefiningOp()) {
    if (!isView(op) || !op->use_empty())
      return;
    value = op->getOperand(0);
    op->erase();
  }
}

// The update's box over dims, counted as shape; none unless it holds
// the update's bytes, so both list them in one order.
std::optional<Window> rebox(const Window &written, llvm::ArrayRef<int64_t> dims,
                            llvm::ArrayRef<int64_t> shape) {
  const Intervals *bytes = written.bytes();
  if (!bytes || bytes->empty() || shape.size() != dims.size())
    return std::nullopt;
  Window box;
  box.dims.assign(dims.begin(), dims.end());
  box.count.assign(shape.begin(), shape.end());
  box.start.resize(dims.size());
  box.stride.assign(dims.size(), 1);
  box.width = written.width;
  // The first element's index, in bounds.
  int64_t offset = bytes->front().first / written.width;
  for (size_t d = dims.size(); d-- > 0;) {
    box.start[d] = offset % dims[d];
    offset /= dims[d];
    if (box.start[d] + box.count[d] > dims[d])
      return std::nullopt;
  }
  if (relate(box, written) != Relation::Equal)
    return std::nullopt;
  return box;
}

// The slice read off the update's tile, or null.
mlir::Value forward(mlir::OpBuilder &builder, SliceOp slice, const Window &window,
                    DynamicUpdateSliceOp update, const Window &written, Relation relation) {
  mlir::Value tile = update.getUpdate();
  mlir::RankedTensorType type = slice.getType();
  mlir::RankedTensorType grid = slice.getOperand().getType();
  builder.setInsertionPoint(slice);
  if (relation == Relation::Equal && type.getElementType() == update.getType().getElementType()) {
    tile = unreshape(tile);
    return type == tile.getType()
               ? tile
               : ReshapeOp::create(builder, slice.getLoc(), type, tile).getResult();
  }
  if (relation != Relation::Inside || grid.getElementType() != update.getType().getElementType())
    return {};

  // The update's box over the slice's operand, the tile in its shape.
  std::optional<Window> box = written;
  if (grid != update.getType()) {
    tile = unreshape(tile);
    box = rebox(written, grid.getShape(),
                mlir::cast<mlir::RankedTensorType>(tile.getType()).getShape());
  }
  if (!box)
    return {};

  // The window shifted to the tile's origin.
  llvm::SmallVector<int64_t> start, limit;
  for (size_t d = 0; d < window.dims.size(); ++d) {
    start.push_back(window.start[d] - box->start[d]);
    limit.push_back(start.back() + (window.count[d] - 1) * window.stride[d] + 1);
  }
  return SliceOp::create(builder, slice.getLoc(), tile, start, limit, window.stride);
}

// The head viewed as the slice's operand, or null.
mlir::Value view(mlir::OpBuilder &builder, SliceOp slice, mlir::Value head) {
  mlir::RankedTensorType grid = slice.getOperand().getType();
  auto type = mlir::cast<mlir::RankedTensorType>(head.getType());
  mlir::Type from = type.getElementType(), to = grid.getElementType();
  // Elements of another bit width: no view.
  if (from != to && (!from.isIntOrFloat() || !to.isIntOrFloat() ||
                     from.getIntOrFloatBitWidth() != to.getIntOrFloatBitWidth()))
    return {};
  builder.setInsertionPoint(slice);
  if (type.getShape() != grid.getShape())
    head = ReshapeOp::create(builder, slice.getLoc(), type.clone(grid.getShape()), head);
  if (from != to)
    head = BitcastConvertOp::create(builder, slice.getLoc(), grid, head);
  return head;
}

struct ForwardSlices : mlir::PassWrapper<ForwardSlices, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForwardSlices)
  llvm::StringRef getArgument() const override {
    return "fsim-forward-slices";
  }
  void runOnOperation() override {
    llvm::SmallVector<SliceOp> slices;
    getOperation().walk([&](SliceOp slice) { slices.push_back(slice); });
    llvm::DenseMap<mlir::Operation *, std::optional<Window>> updates;
    mlir::OpBuilder builder(&getContext());

    for (SliceOp slice : slices) {
      std::optional<Window> window = Window::of(slice);
      if (!window)
        continue;
      mlir::Value value = slice.getOperand();
      bool missed = false;
      for (int hop = 0; hop < kHops; ++hop) {
        mlir::Operation *op = value.getDefiningOp();
        if (op && isView(op)) {
          value = op->getOperand(0);
          continue;
        }
        auto update = mlir::dyn_cast_if_present<DynamicUpdateSliceOp>(op);
        // Past every update: read the chain's head, when it has a view.
        if (!update) {
          mlir::Value head = missed ? view(builder, slice, value) : mlir::Value();
          if (head) {
            mlir::Value operand = slice.getOperand();
            slice.getOperandMutable().assign(head);
            drop(operand);
          }
          break;
        }
        auto [it, fresh] = updates.try_emplace(op);
        if (fresh)
          it->second = Window::of(update);
        if (!it->second)
          break;
        const Window &written = *it->second;
        Relation relation = relate(*window, written);
        // A missed update: read what it overwrote.
        if (relation == Relation::Disjoint) {
          value = update.getOperand();
          missed = true;
          continue;
        }
        if (mlir::Value forwarded = forward(builder, slice, *window, update, written, relation)) {
          mlir::Value operand = slice.getOperand();
          slice.getResult().replaceAllUsesWith(forwarded);
          slice.erase();
          drop(operand);
        }
        break;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createForwardSlicesPass() {
  return std::make_unique<ForwardSlices>();
}

} // namespace fsim
