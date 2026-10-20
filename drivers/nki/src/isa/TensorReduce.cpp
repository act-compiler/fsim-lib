// tensor_reduce_arith: the last num_r_dim dims reduced in f32, maybe negated.
#include <stdexcept>

#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

std::string op(Fields &f) {
  std::string name = f.word("op");
  if (name != "add" && name != "multiply" && name != "maximum" && name != "minimum")
    f.fail("op", "'" + name + "' is not decoded in M1");
  return name;
}

int64_t dims(Fields &f) {
  Number n = f.number("num_r_dim");
  if (!n.integer || *n.integer < 0)
    f.fail("num_r_dim", "is not a count");
  return *n.integer;
}

class TensorReduceArith final : public Instruction {
public:
  explicit TensorReduceArith(Fields &f)
      : dst_(f.view("dst")), op_(op(f)), data_(f.view("data")), dims_(dims(f)),
        negate_(f.flag("negate")) {
    // Flat-order writes ignore kept unit dims.
    f.skip("keepdims");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int x = module.toF32(module.read(data_));
    int64_t rank = static_cast<int64_t>(module.shape(x).size());
    if (dims_ > rank)
      throw std::runtime_error("tensor_reduce_arith reduces " + std::to_string(dims_) +
                               " dims of a rank " + std::to_string(rank) + " operand");
    std::vector<int64_t> axes;
    for (int64_t i = rank - dims_; i < rank; ++i)
      axes.push_back(i);
    int result = axes.empty() ? x : module.reduce(x, op_, axes);
    if (negate_)
      result = module.unary("negate", result);
    module.write(dst_, module.toDtype(module.reshape(result, dst_.shape()), dst_.dtype));
    accelerator.wrote(dst_);
  }

private:
  const View dst_;
  const std::string op_;
  const View data_;
  const int64_t dims_;
  const bool negate_;
};

} // namespace

InstructionPtr decodeTensorReduceArith(Fields &fields) {
  return std::make_unique<TensorReduceArith>(fields);
}

} // namespace nki
