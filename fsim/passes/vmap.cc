// Vmaps @main at batch size B: each argument and result takes
// a leading batch dimension of size B. Each op's rule batches it in place,
// as the rule's comment says; an op with no rule is refused by name.
#include "fsim_passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"

namespace fsim {
namespace {

namespace hlo = mlir::stablehlo;

// The rewrite's state: the batch size and which values are batched.
struct Batching {
  explicit Batching(int64_t batch_size, mlir::MLIRContext *context)
      : batch_size(batch_size), builder(context) {
  }

  // The type with the batch dimension in front.
  mlir::RankedTensorType addBatchDim(mlir::Type type) {
    auto ranked = mlir::cast<mlir::RankedTensorType>(type);
    llvm::SmallVector<int64_t> shape{batch_size};
    shape.append(ranked.getShape().begin(), ranked.getShape().end());
    return ranked.clone(shape);
  }

  // Gives op's results the batch dimension.
  void addBatchDim(mlir::Operation *op) {
    for (mlir::Value result : op->getResults()) {
      result.setType(addBatchDim(result.getType()));
      batched.insert(result);
    }
  }

  bool isBatched(mlir::Value value) const {
    return batched.contains(value);
  }

  // The value with the batch dimension, an unbatched one broadcast once.
  mlir::Value broadcast(mlir::Value value) {
    if (isBatched(value))
      return value;
    auto [it, fresh] = broadcasts.try_emplace(value);
    if (fresh) {
      // Operand dimension d lands on d + 1.
      auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
      auto dims = llvm::to_vector(llvm::seq<int64_t>(1, type.getRank() + 1));
      builder.setInsertionPointAfterValue(value);
      it->second =
          hlo::BroadcastInDimOp::create(builder, value.getLoc(), addBatchDim(type), value, dims);
      batched.insert(it->second);
    }
    return it->second;
  }

  // Batches op's operand i.
  void batchOperand(mlir::Operation *op, unsigned i) {
    op->setOperand(i, broadcast(op->getOperand(i)));
  }

  int64_t batch_size;
  mlir::OpBuilder builder;
  llvm::DenseSet<mlir::Value> batched;
  // Each unbatched value's broadcast along the batch dimension.
  llvm::DenseMap<mlir::Value, mlir::Value> broadcasts;
};

// A rule batches one op in place; a failure refuses it.
using Rule = mlir::LogicalResult (*)(Batching &, mlir::Operation *);

// Elementwise: a batched operand batches every operand and result.
mlir::LogicalResult elementwise(Batching &batching, mlir::Operation *op) {
  if (llvm::none_of(op->getOperands(), [&](mlir::Value v) { return batching.isBatched(v); }))
    return mlir::success();
  auto type = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    mlir::Value operand = op->getOperand(i);
    int64_t rank = mlir::cast<mlir::RankedTensorType>(operand.getType()).getRank();
    if (rank - batching.isBatched(operand) == type.getRank()) {
      batching.batchOperand(op, i);
      continue;
    }
    // A scalar beside tensors, e.g. select's pred: unbatched stays scalar.
    if (!batching.isBatched(operand))
      continue;
    // A batched scalar spreads over the result's shape.
    auto element = mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();
    batching.builder.setInsertionPoint(op);
    op->setOperand(i, hlo::BroadcastInDimOp::create(batching.builder, op->getLoc(),
                                                    batching.addBatchDim(type.clone(element)),
                                                    operand, llvm::ArrayRef<int64_t>{0}));
  }
  batching.addBatchDim(op);
  return mlir::success();
}

// A constant stays unbatched.
mlir::LogicalResult constant(Batching &, mlir::Operation *) {
  return mlir::success();
}

// Operand dimensions shift by one; a batched operand's batch dimension maps to 0.
mlir::LogicalResult broadcastInDim(Batching &batching, mlir::Operation *op) {
  auto broadcast = mlir::cast<hlo::BroadcastInDimOp>(op);
  llvm::SmallVector<int64_t> dims;
  if (batching.isBatched(broadcast.getOperand()))
    dims.push_back(0);
  for (int64_t d : broadcast.getBroadcastDimensions())
    dims.push_back(d + 1);
  broadcast.setBroadcastDimensions(dims);
  batching.addBatchDim(op);
  return mlir::success();
}

// Batched operand, batched result.
mlir::LogicalResult reshape(Batching &batching, mlir::Operation *op) {
  batching.batchOperand(op, 0);
  batching.addBatchDim(op);
  return mlir::success();
}

// The batch dimension stays first; the permutation shifts by one.
mlir::LogicalResult transpose(Batching &batching, mlir::Operation *op) {
  auto transpose = mlir::cast<hlo::TransposeOp>(op);
  llvm::SmallVector<int64_t> permutation{0};
  for (int64_t d : transpose.getPermutation())
    permutation.push_back(d + 1);
  transpose.setPermutation(permutation);
  batching.batchOperand(op, 0);
  batching.addBatchDim(op);
  return mlir::success();
}

// The dimensions shifted by one.
llvm::SmallVector<int64_t> shifted(llvm::ArrayRef<int64_t> dims) {
  llvm::SmallVector<int64_t> result;
  for (int64_t d : dims)
    result.push_back(d + 1);
  return result;
}

// The values with the batch dimension's entry in front.
llvm::SmallVector<int64_t> leading(int64_t first, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<int64_t> result{first};
  result.append(values.begin(), values.end());
  return result;
}

// A 0-d operand must stay unbatched; a batched one is refused.
mlir::LogicalResult scalar(Batching &batching, mlir::Operation *op, mlir::ValueRange values,
                           llvm::StringRef role) {
  for (mlir::Value value : values)
    if (batching.isBatched(value))
      return mlir::emitError(op->getLoc())
             << "no vmap rule for a batched " << role << " of " << op->getName();
  return mlir::success();
}

// The batch dimension slices whole: start 0, limit B, stride 1.
mlir::LogicalResult slice(Batching &batching, mlir::Operation *op) {
  auto slice = mlir::cast<hlo::SliceOp>(op);
  slice.setStartIndices(leading(0, slice.getStartIndices()));
  slice.setLimitIndices(leading(batching.batch_size, slice.getLimitIndices()));
  slice.setStrides(leading(1, slice.getStrides()));
  batching.batchOperand(op, 0);
  batching.addBatchDim(op);
  return mlir::success();
}

// The scalar start indices take a leading 0 for the batch dimension.
mlir::LogicalResult dynamicUpdateSlice(Batching &batching, mlir::Operation *op) {
  auto update = mlir::cast<hlo::DynamicUpdateSliceOp>(op);
  if (mlir::failed(scalar(batching, op, update.getStartIndices(), "start index")))
    return mlir::failure();
  batching.batchOperand(op, 0);
  batching.batchOperand(op, 1);
  mlir::Type index = update.getStartIndices().front().getType();
  batching.builder.setInsertionPoint(op);
  mlir::Value zero =
      hlo::ConstantOp::create(batching.builder, op->getLoc(),
                              mlir::cast<mlir::ElementsAttr>(batching.builder.getZeroAttr(index)));
  llvm::SmallVector<mlir::Value> operands{op->getOperand(0), op->getOperand(1), zero};
  operands.append(update.getStartIndices().begin(), update.getStartIndices().end());
  op->setOperands(operands);
  batching.addBatchDim(op);
  return mlir::success();
}

// The batch dimension is a leading batching dimension of both operands.
mlir::LogicalResult dotGeneral(Batching &batching, mlir::Operation *op) {
  auto dot = mlir::cast<hlo::DotGeneralOp>(op);
  hlo::DotDimensionNumbersAttr numbers = dot.getDotDimensionNumbers();
  dot.setDotDimensionNumbersAttr(hlo::DotDimensionNumbersAttr::get(
      op->getContext(), leading(0, shifted(numbers.getLhsBatchingDimensions())),
      leading(0, shifted(numbers.getRhsBatchingDimensions())),
      shifted(numbers.getLhsContractingDimensions()),
      shifted(numbers.getRhsContractingDimensions())));
  batching.batchOperand(op, 0);
  batching.batchOperand(op, 1);
  batching.addBatchDim(op);
  return mlir::success();
}

// The batch dimension slices whole into result dimension 0; batched indices are refused.
mlir::LogicalResult gather(Batching &batching, mlir::Operation *op) {
  auto gather = mlir::cast<hlo::GatherOp>(op);
  if (batching.isBatched(gather.getStartIndices()))
    return mlir::emitError(op->getLoc()) << "no vmap rule for batched indices of " << op->getName();
  batching.batchOperand(op, 0);
  hlo::GatherDimensionNumbersAttr numbers = gather.getDimensionNumbers();
  gather.setDimensionNumbersAttr(hlo::GatherDimensionNumbersAttr::get(
      op->getContext(), leading(0, shifted(numbers.getOffsetDims())),
      shifted(numbers.getCollapsedSliceDims()), shifted(numbers.getOperandBatchingDims()),
      numbers.getStartIndicesBatchingDims(), shifted(numbers.getStartIndexMap()),
      numbers.getIndexVectorDim()));
  gather.setSliceSizes(leading(batching.batch_size, gather.getSliceSizes()));
  batching.addBatchDim(op);
  return mlir::success();
}

// The reduced dimensions shift; the scalar inits and the body stay.
mlir::LogicalResult reduce(Batching &batching, mlir::Operation *op) {
  auto reduce = mlir::cast<hlo::ReduceOp>(op);
  if (mlir::failed(scalar(batching, op, reduce.getInitValues(), "init value")))
    return mlir::failure();
  reduce.setDimensions(shifted(reduce.getDimensions()));
  for (unsigned i = 0; i < reduce.getInputs().size(); ++i)
    batching.batchOperand(op, i);
  batching.addBatchDim(op);
  return mlir::success();
}

// Each result batched as its operand.
mlir::LogicalResult barrier(Batching &batching, mlir::Operation *op) {
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    if (batching.isBatched(op->getOperand(i))) {
      op->getResult(i).setType(op->getOperand(i).getType());
      batching.batched.insert(op->getResult(i));
    }
  return mlir::success();
}

// Every result batched.
mlir::LogicalResult ret(Batching &batching, mlir::Operation *op) {
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    batching.batchOperand(op, i);
  return mlir::success();
}

// The rules by op name.
const std::pair<llvm::StringLiteral, Rule> kRules[] = {
    {hlo::SelectOp::getOperationName(), elementwise},
    {hlo::ConstantOp::getOperationName(), constant},
    {hlo::BroadcastInDimOp::getOperationName(), broadcastInDim},
    {hlo::ReshapeOp::getOperationName(), reshape},
    {hlo::TransposeOp::getOperationName(), transpose},
    {hlo::SliceOp::getOperationName(), slice},
    {hlo::DynamicUpdateSliceOp::getOperationName(), dynamicUpdateSlice},
    {hlo::BitcastConvertOp::getOperationName(), reshape},
    {hlo::DotGeneralOp::getOperationName(), dotGeneral},
    {hlo::GatherOp::getOperationName(), gather},
    {hlo::ReduceOp::getOperationName(), reduce},
    {hlo::OptimizationBarrierOp::getOperationName(), barrier},
    {mlir::func::ReturnOp::getOperationName(), ret},
};

// Whether every type is a ranked tensor.
bool ranked(mlir::TypeRange types) {
  return llvm::all_of(types,
                      [](mlir::Type type) { return mlir::isa<mlir::RankedTensorType>(type); });
}

// The op's rule, or null.
Rule rule(mlir::Operation *op) {
  if (op->hasTrait<mlir::OpTrait::Elementwise>() &&
      mlir::isa<hlo::StablehloDialect>(op->getDialect()) && ranked(op->getOperandTypes()) &&
      ranked(op->getResultTypes()))
    return elementwise;
  for (const auto &[name, rule] : kRules)
    if (op->getName().getStringRef() == name)
      return rule;
  return nullptr;
}

struct Vmap : mlir::PassWrapper<Vmap, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Vmap)
  explicit Vmap(int64_t batch_size) : batch_size(batch_size) {
  }
  llvm::StringRef getArgument() const override {
    return "fsim-vmap";
  }
  void runOnOperation() override {
    // The library phase already refused a module without @main.
    auto main = getOperation().lookupSymbol<mlir::func::FuncOp>("main");
    Batching batching(batch_size, &getContext());
    mlir::Block &body = main.getBody().front();
    for (mlir::BlockArgument argument : body.getArguments()) {
      argument.setType(batching.addBatchDim(argument.getType()));
      batching.batched.insert(argument);
    }

    // The ops in order, before the rules add broadcasts.
    auto ops = llvm::map_to_vector(body, [](mlir::Operation &op) { return &op; });
    for (mlir::Operation *op : ops) {
      Rule opRule = rule(op);
      if (!opRule) {
        mlir::emitError(op->getLoc()) << "no vmap rule for " << op->getName();
        return signalPassFailure();
      }
      if (mlir::failed(opRule(batching, op)))
        return signalPassFailure();
    }
    main.setType(mlir::FunctionType::get(&getContext(), body.getArgumentTypes(),
                                         body.getTerminator()->getOperandTypes()));
  }

  int64_t batch_size;
};

} // namespace

std::unique_ptr<mlir::Pass> createVmapPass(int64_t batch_size) {
  return std::make_unique<Vmap>(batch_size);
}

} // namespace fsim
