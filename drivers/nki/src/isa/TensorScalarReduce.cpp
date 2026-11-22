// tensor_scalar_reduce: data op0 operand0, reduced into the Vector Engine's accumulator registers.
#include <set>

#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

std::string reduceOp(Fields &f) {
  static const std::set<std::string> ops = {"add", "multiply", "maximum", "minimum"};
  std::string name = f.word("reduce_op");
  if (!ops.count(name))
    f.fail("reduce_op", "'" + name + "' is not a documented reduce op");
  return name;
}

class TensorScalarReduce final : public Instruction {
public:
  explicit TensorScalarReduce(Fields &f)
      : dst_(f.view("dst")), data_(f.view("data")), op0_(scalarOp(f)),
        operand0_(f.operand("operand0")), reduceOp_(reduceOp(f)),
        reduceRes_(f.optView("reduce_res")) {
    if (std::holds_alternative<None>(operand0_))
      f.fail("operand0", "is none");
    // No reverse, and reset_reduce only, are decoded.
    f.refuse("reverse0");
    f.expect("reduce_cmd", "reset_reduce");
    f.expectNone("reduce_init");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int result = scalarArith(module, dst_, data_, op0_, operand0_);
    module.write(dst_, module.toDtype(result, dst_.dtype));
    accelerator.wrote(dst_);
    // reset_reduce stores dst's reduction, as read back.
    int reduced = module.toF32(module.read(dst_));
    std::vector<int64_t> dims = module.shape(reduced), axes;
    for (size_t i = 1; i < dims.size(); ++i)
      axes.push_back(static_cast<int64_t>(i));
    if (!axes.empty())
      reduced = module.reduce(reduced, reduceOp_, axes);
    module.writeLanes(Slot::accumulatorRegs, data_.partition,
                      module.reshape(reduced, {dims[0], 1}));
    // reduce_res reads the lanes from its own partition.
    if (reduceRes_) {
      const View &res = *reduceRes_;
      int regs = module.readLanes(Slot::accumulatorRegs, res.partition, res.shape()[0]);
      module.write(res, module.toDtype(module.reshape(regs, res.shape()), res.dtype));
      accelerator.wrote(res);
    }
  }

private:
  const View dst_, data_;
  const std::string op0_;
  const Value operand0_;
  const std::string reduceOp_;
  const std::optional<View> reduceRes_;
};

} // namespace

InstructionPtr decodeTensorScalarReduce(Fields &fields) {
  return std::make_unique<TensorScalarReduce>(fields);
}

} // namespace nki
