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

// A constant stays unbatched.
mlir::LogicalResult constant(Batching &, mlir::Operation *) {
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
    {hlo::ConstantOp::getOperationName(), constant},
    {mlir::func::ReturnOp::getOperationName(), ret},
};

// The op's rule, or null.
Rule rule(mlir::Operation *op) {
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
