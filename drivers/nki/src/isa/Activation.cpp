// activation: op(data * scale + bias) and the reduce registers.
#include <map>

#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

enum class Fn { copy, exp, log, sqrt, rsqrt, tanh, sigmoid, relu, square, abs, reciprocal };
enum class Cmd { idle, reset, reduce, resetReduce, loadReduce };

Fn fn(Fields &f) {
  static const std::map<std::string, Fn> fns = {{"copy", Fn::copy},
                                                {"exp", Fn::exp},
                                                {"log", Fn::log},
                                                {"sqrt", Fn::sqrt},
                                                {"rsqrt", Fn::rsqrt},
                                                {"tanh", Fn::tanh},
                                                {"sigmoid", Fn::sigmoid},
                                                {"relu", Fn::relu},
                                                {"square", Fn::square},
                                                {"abs", Fn::abs},
                                                {"reciprocal", Fn::reciprocal}};
  std::string op = f.word("op");
  auto it = fns.find(op);
  if (it == fns.end())
    f.fail("op", "'" + op + "' is not decoded in M1");
  return it->second;
}

Cmd cmd(Fields &f) {
  static const std::map<std::string, Cmd> cmds = {{"idle", Cmd::idle},
                                                  {"reset", Cmd::reset},
                                                  {"reduce", Cmd::reduce},
                                                  {"reset_reduce", Cmd::resetReduce},
                                                  {"load_reduce", Cmd::loadReduce}};
  std::string word = f.word("reduce_cmd");
  auto it = cmds.find(word);
  if (it == cmds.end())
    f.fail("reduce_cmd", "'" + word + "' is not a reduce command");
  return it->second;
}

std::optional<std::string> reduceOp(Fields &f) {
  std::optional<std::string> op = f.optWord("reduce_op");
  if (op && *op != "add" && *op != "multiply" && *op != "maximum" && *op != "minimum")
    f.fail("reduce_op", "'" + *op + "' is not decoded in M1");
  return op;
}

class Activation final : public Instruction {
public:
  explicit Activation(Fields &f)
      : dst_(f.view("dst")), fn_(fn(f)), data_(f.view("data")), bias_(f.operand("bias")),
        scale_(f.operand("scale")), reduceOp_(reduceOp(f)), reduceRes_(f.optView("reduce_res")),
        cmd_(cmd(f)) {
    if (std::holds_alternative<None>(scale_))
      f.fail("scale", "is none");
    bool reduces = cmd_ == Cmd::reduce || cmd_ == Cmd::resetReduce;
    if (cmd_ == Cmd::idle && reduceOp_)
      f.fail("reduce_op", "is set while reduce_cmd is idle");
    if (reduces && !reduceOp_)
      f.fail("reduce_op", "is none while reduce_cmd reduces");
    // The ISA reduces into reduce_regs by add only.
    if (reduces && *reduceOp_ != "add")
      f.fail("reduce_op", "'" + *reduceOp_ + "' with reduce_cmd " +
                              (cmd_ == Cmd::reduce ? "reduce" : "reset_reduce") +
                              "; activation reduces only by add");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    const View &dst = dst_;
    int x = module.toF32(module.read(data_));
    if (data_.count() == dst.count())
      x = module.reshape(x, dst.shape());
    // Scale, unless a scalar 1.0; then bias, unless none.
    if (!std::holds_alternative<Number>(scale_) || std::get<Number>(scale_).value != 1.0)
      x = apply(module, "multiply", x, scale_);
    if (!std::holds_alternative<None>(bias_))
      x = apply(module, "add", x, bias_);
    int result = compute(module, x);
    if (module.type(result).count() == dst.count())
      result = module.reshape(result, dst.shape());
    module.write(dst, module.toDtype(result, dst.dtype));
    accelerator.wrote(dst);
    accumulate(accelerator, module, result);
  }

private:
  // x op operand: a scalar, or a unit-padded view.
  int apply(Module &module, const std::string &op, int x, const Value &operand) const {
    std::vector<int64_t> dims = module.shape(x);
    if (auto *n = std::get_if<Number>(&operand))
      return module.binary(op, x, module.f32(n->value, dims));
    int v = module.toF32(module.read(std::get<View>(operand)));
    std::vector<int64_t> padded = module.shape(v);
    if (padded.size() < dims.size())
      padded.resize(dims.size(), 1);
    auto [lhs, rhs] = module.broadcastPair(x, module.reshape(v, padded));
    return module.binary(op, lhs, rhs);
  }

  int compute(Module &module, int x) const {
    std::vector<int64_t> dims = module.shape(x);
    switch (fn_) {
    case Fn::copy:
      return x;
    case Fn::exp:
      return module.unary("exponential", x);
    case Fn::log:
      return module.unary("log", x);
    case Fn::sqrt:
      return module.unary("sqrt", x);
    case Fn::rsqrt:
      return module.binary("divide", module.f32(1.0, dims), module.unary("sqrt", x));
    case Fn::tanh:
      return module.unary("tanh", x);
    case Fn::sigmoid:
      return module.unary("logistic", x);
    case Fn::relu:
      return module.binary("maximum", x, module.f32(0.0, dims));
    case Fn::square:
      return module.binary("multiply", x, x);
    case Fn::abs:
      return module.unary("abs", x);
    case Fn::reciprocal:
      return module.binary("divide", module.f32(1.0, dims), x);
    }
    return x;
  }

  // Reduce register lanes from data's partition: reset, reduce, read.
  void accumulate(Accelerator &accelerator, Module &module, int result) const {
    std::vector<int64_t> dims = module.shape(result);
    int64_t first = data_.partition, lanes = dims[0];
    if (cmd_ == Cmd::reset)
      module.writeLanes(Slot::reduceRegs, first, module.f32(0.0, {lanes, 1}));
    if (cmd_ == Cmd::reduce || cmd_ == Cmd::resetReduce) {
      std::vector<int64_t> axes;
      for (size_t i = 1; i < dims.size(); ++i)
        axes.push_back(static_cast<int64_t>(i));
      int reduced = axes.empty() ? result : module.reduce(result, *reduceOp_, axes);
      reduced = module.reshape(reduced, {lanes, 1});
      // reset_reduce stores it; reduce adds it on.
      if (cmd_ == Cmd::reduce)
        reduced =
            module.binary(*reduceOp_, module.readLanes(Slot::reduceRegs, first, lanes), reduced);
      module.writeLanes(Slot::reduceRegs, first, reduced);
    }
    if (reduceRes_) {
      const View &res = *reduceRes_;
      int regs = module.readLanes(Slot::reduceRegs, first, lanes);
      module.write(res, module.toDtype(module.reshape(regs, res.shape()), res.dtype));
      accelerator.wrote(res);
    }
  }

  const View dst_;
  const Fn fn_;
  const View data_;
  const Value bias_, scale_;
  const std::optional<std::string> reduceOp_;
  const std::optional<View> reduceRes_;
  const Cmd cmd_;
};

} // namespace

InstructionPtr decodeActivation(Fields &fields) {
  return std::make_unique<Activation>(fields);
}

} // namespace nki
