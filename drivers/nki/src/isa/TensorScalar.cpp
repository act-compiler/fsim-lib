// tensor_scalar_arith: data op0 operand0, in f32.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class TensorScalarArith final : public Instruction {
public:
  explicit TensorScalarArith(Fields &f)
      : dst_(f.view("dst")), data_(f.view("data")), op0_(scalarOp(f)),
        operand0_(f.operand("operand0")) {
    if (std::holds_alternative<None>(operand0_))
      f.fail("operand0", "is none");
    // Neither reverse nor a second op is decoded.
    f.refuse("reverse0");
    f.expectNone("op1");
    f.expectNone("operand1");
    f.refuse("reverse1");
    f.skip("engine");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int result = scalarArith(module, dst_, data_, op0_, operand0_);
    module.write(dst_, module.toDtype(result, dst_.dtype));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, data_;
  const std::string op0_;
  const Value operand0_;
};

} // namespace

InstructionPtr decodeTensorScalarArith(Fields &fields) {
  return std::make_unique<TensorScalarArith>(fields);
}

} // namespace nki
