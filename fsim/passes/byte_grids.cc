// Normalizes byte tensors to byte grids: a bitcast between a [..., K] byte
// grid and K-byte elements sinks below the slices, reshapes and updates
// that consume it, so byte tensors stay bytes and bitcasts act on tiles;
// an update of a 2-D byte tensor's [P, M, K] grid runs in the byte
// tensor's own shape.
#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace fsim {
namespace {

using mlir::stablehlo::BitcastConvertOp;

// Whether an element is one byte.
bool isByte(mlir::Type type) {
  return type.isIntOrFloat() && type.getIntOrFloatBitWidth() == 8;
}

// The bitcast from a [..., K] byte grid to K-byte elements, or null.
BitcastConvertOp narrowing(mlir::Value value) {
  auto op = value.getDefiningOp<BitcastConvertOp>();
  if (!op)
    return {};
  mlir::RankedTensorType from = op.getOperand().getType();
  return from.getRank() == op.getType().getRank() + 1 && isByte(from.getElementType()) ? op
                                                                                       : nullptr;
}

// The bitcast from K-byte elements to a [..., K] byte grid, or null.
BitcastConvertOp widening(mlir::Value value) {
  auto op = value.getDefiningOp<BitcastConvertOp>();
  if (!op)
    return {};
  mlir::RankedTensorType to = op.getType();
  return to.getRank() == op.getOperand().getType().getRank() + 1 && isByte(to.getElementType())
             ? op
             : nullptr;
}

// The shape with a trailing dim of k appended.
llvm::SmallVector<int64_t> append(llvm::ArrayRef<int64_t> shape, int64_t k) {
  llvm::SmallVector<int64_t> out(shape);
  out.push_back(k);
  return out;
}

// slice(narrow(y)) -> narrow(slice(y)), over the whole byte dim.
struct SliceOfNarrow : mlir::OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp slice,
                                      mlir::PatternRewriter &rewriter) const override {
    BitcastConvertOp narrow = narrowing(slice.getOperand());
    if (!narrow)
      return mlir::failure();
    mlir::TypedValue<mlir::RankedTensorType> grid = narrow.getOperand();
    int64_t k = grid.getType().getShape().back();
    auto bytes = mlir::stablehlo::SliceOp::create(
        rewriter, slice.getLoc(), grid, append(slice.getStartIndices(), 0),
        append(slice.getLimitIndices(), k), append(slice.getStrides(), 1));
    rewriter.replaceOpWithNewOp<BitcastConvertOp>(slice, slice.getType(), bytes);
    return mlir::success();
  }
};

// slice(widen(x)) -> widen(slice(x)) when it keeps the byte dim whole.
struct SliceOfWiden : mlir::OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp slice,
                                      mlir::PatternRewriter &rewriter) const override {
    BitcastConvertOp widen = widening(slice.getOperand());
    if (!widen || slice.getStartIndices().back() != 0 || slice.getStrides().back() != 1 ||
        slice.getType().getShape().back() != widen.getType().getShape().back())
      return mlir::failure();
    auto elements = mlir::stablehlo::SliceOp::create(
        rewriter, slice.getLoc(), widen.getOperand(), slice.getStartIndices().drop_back(),
        slice.getLimitIndices().drop_back(), slice.getStrides().drop_back());
    rewriter.replaceOpWithNewOp<BitcastConvertOp>(slice, slice.getType(), elements);
    return mlir::success();
  }
};

// reshape(narrow(y)) -> narrow(reshape(y)); reshape(widen(x)) ->
// widen(reshape(x)) when the reshape keeps the byte dim whole.
struct ReshapeOfBitcast : mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp reshape,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType type = reshape.getType();
    if (BitcastConvertOp narrow = narrowing(reshape.getOperand())) {
      mlir::RankedTensorType grid = narrow.getOperand().getType();
      auto bytes = mlir::stablehlo::ReshapeOp::create(
          rewriter, reshape.getLoc(), grid.clone(append(type.getShape(), grid.getShape().back())),
          narrow.getOperand());
      rewriter.replaceOpWithNewOp<BitcastConvertOp>(reshape, type, bytes);
      return mlir::success();
    }
    BitcastConvertOp widen = widening(reshape.getOperand());
    if (!widen || type.getRank() == 0 ||
        type.getShape().back() != widen.getType().getShape().back())
      return mlir::failure();
    mlir::RankedTensorType from = widen.getOperand().getType();
    auto elements = mlir::stablehlo::ReshapeOp::create(
        rewriter, reshape.getLoc(), from.clone(type.getShape().drop_back()), widen.getOperand());
    rewriter.replaceOpWithNewOp<BitcastConvertOp>(reshape, type, elements);
    return mlir::success();
  }
};

// dus(narrow(y), u, i...) -> narrow(dus(y, widen(u), i..., 0)).
struct UpdateOfNarrow : mlir::OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp update,
                                      mlir::PatternRewriter &rewriter) const override {
    BitcastConvertOp narrow = narrowing(update.getOperand());
    if (!narrow || update.getStartIndices().empty())
      return mlir::failure();
    mlir::TypedValue<mlir::RankedTensorType> grid = narrow.getOperand();
    mlir::RankedTensorType tile = update.getUpdate().getType();
    auto bytes = BitcastConvertOp::create(
        rewriter, update.getLoc(),
        grid.getType().clone(append(tile.getShape(), grid.getType().getShape().back())),
        update.getUpdate());
    // The byte dim's start index, a zero of the others' type.
    llvm::SmallVector<mlir::Value> indices(update.getStartIndices());
    indices.push_back(mlir::stablehlo::ConstantOp::create(
        rewriter, update.getLoc(), rewriter.getZeroAttr(indices.front().getType())));
    auto written = mlir::stablehlo::DynamicUpdateSliceOp::create(rewriter, update.getLoc(), grid,
                                                                 bytes, indices);
    rewriter.replaceOpWithNewOp<BitcastConvertOp>(update, update.getType(), written);
    return mlir::success();
  }
};

// dus(reshape(x), u, i, j, 0) -> reshape(dus(x, reshape(u), i, j * K)),
// x a [P, M * K] byte grid and u a [p, n, K] tile, so updates keep x's shape.
struct UpdateOfGrid : mlir::OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp update,
                                      mlir::PatternRewriter &rewriter) const override {
    auto reshape = update.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!reshape)
      return mlir::failure();
    mlir::TypedValue<mlir::RankedTensorType> x = reshape.getOperand();
    llvm::ArrayRef<int64_t> from = x.getType().getShape(), grid = update.getType().getShape(),
                            tile = update.getUpdate().getType().getShape();
    mlir::OperandRange indices = update.getStartIndices();
    if (from.size() != 2 || grid.size() != 3 || from[0] != grid[0] || tile[2] != grid[2] ||
        !isByte(x.getType().getElementType()) || !mlir::matchPattern(indices[2], mlir::m_Zero()))
      return mlir::failure();
    int64_t k = grid[2];
    auto row = mlir::stablehlo::ReshapeOp::create(
        rewriter, update.getLoc(), update.getUpdate().getType().clone({tile[0], tile[1] * k}),
        update.getUpdate());
    // The column index scaled to bytes, folded when constant.
    auto index = mlir::cast<mlir::RankedTensorType>(indices[1].getType());
    mlir::DenseIntElementsAttr value;
    mlir::Value column;
    if (mlir::matchPattern(indices[1], mlir::m_Constant(&value))) {
      column = mlir::stablehlo::ConstantOp::create(
          rewriter, update.getLoc(), mlir::DenseElementsAttr::get(index, *value.begin() * k));
    } else {
      auto scale = mlir::stablehlo::ConstantOp::create(
          rewriter, update.getLoc(),
          mlir::DenseElementsAttr::get(index, rewriter.getIntegerAttr(index.getElementType(), k)));
      column = mlir::stablehlo::MulOp::create(rewriter, update.getLoc(), indices[1], scale);
    }
    auto written = mlir::stablehlo::DynamicUpdateSliceOp::create(
        rewriter, update.getLoc(), x, row, mlir::ValueRange{indices[0], column});
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(update, update.getType(), written);
    return mlir::success();
  }
};

// bitcast(bitcast(x)) -> x when it returns to x's type.
struct BitcastOfBitcast : mlir::OpRewritePattern<BitcastConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(BitcastConvertOp outer,
                                      mlir::PatternRewriter &rewriter) const override {
    auto inner = outer.getOperand().getDefiningOp<BitcastConvertOp>();
    if (!inner || inner.getOperand().getType() != outer.getType())
      return mlir::failure();
    rewriter.replaceOp(outer, inner.getOperand());
    return mlir::success();
  }
};

struct ByteGrids : mlir::PassWrapper<ByteGrids, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ByteGrids)
  llvm::StringRef getArgument() const override {
    return "fsim-byte-grids";
  }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SliceOfNarrow, SliceOfWiden, ReshapeOfBitcast, UpdateOfNarrow, UpdateOfGrid,
                 BitcastOfBitcast>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createByteGridsPass() {
  return std::make_unique<ByteGrids>();
}

} // namespace fsim
