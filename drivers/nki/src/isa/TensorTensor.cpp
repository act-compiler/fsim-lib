// tensor_tensor_arith: data1 op data2, in f32 or gpsimd integers.
#include <set>

#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

std::string op(Fields &f, bool native) {
  static const std::set<std::string> ops = {"add",     "subtract", "multiply", "divide",
                                            "maximum", "minimum",  "power"};
  static const std::set<std::string> integer = {"add", "subtract", "multiply", "maximum",
                                                "minimum"};
  std::string name = f.word("op");
  if (!ops.count(name) && !compares().count(name))
    f.fail("op", "'" + name + "' is not decoded");
  if (native && !integer.count(name))
    f.fail("op", "'" + name + "' on integers on gpsimd is not decoded");
  return name;
}

class TensorTensorArith final : public Instruction {
public:
  explicit TensorTensorArith(Fields &f)
      : dst_(f.view("dst")), data1_(f.view("data1")), data2_(f.view("data2")),
        engine_(f.word("engine")),
        native_(engine_ == "gpsimd" && integral(data1_.dtype) && data1_.dtype == data2_.dtype),
        op_(op(f, native_)) {
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int a = module.read(data1_);
    int b = module.read(data2_);
    if (!native_) {
      a = module.toF32(a);
      b = module.toF32(b);
    }
    // Equal counts reshape to data1's shape; else numpy broadcasting.
    if (module.shape(a) != module.shape(b) && module.type(a).count() == module.type(b).count())
      b = module.reshape(b, module.shape(a));
    auto [lhs, rhs] = module.broadcastPair(a, b);
    auto compare = compares().find(op_);
    int result =
        compare == compares().end()
            ? module.binary(op_, lhs, rhs)
            : module.select(module.compare(compare->second, lhs, rhs),
                            module.f32(1, module.shape(lhs)), module.f32(0, module.shape(lhs)));
    // A larger result keeps its first elements, in flat order.
    int64_t n = module.type(result).count();
    if (n > dst_.count())
      result = module.slice(module.reshape(result, {n}), {0}, {dst_.count()}, {1});
    module.write(dst_, module.toDtype(module.reshape(result, dst_.shape()), dst_.dtype));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, data1_, data2_;
  const std::string engine_;
  const bool native_;
  const std::string op_;
};

} // namespace

InstructionPtr decodeTensorTensorArith(Fields &fields) {
  return std::make_unique<TensorTensorArith>(fields);
}

} // namespace nki
