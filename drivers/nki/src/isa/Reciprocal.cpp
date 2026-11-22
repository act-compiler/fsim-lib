// reciprocal: 1 / data, in f32.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class Reciprocal final : public Instruction {
public:
  explicit Reciprocal(Fields &f) : dst_(f.view("dst")), data_(f.view("data")) {
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int x = module.reshape(module.toF32(module.read(data_)), dst_.shape());
    int result = module.binary("divide", module.f32(1.0, dst_.shape()), x);
    module.write(dst_, module.toDtype(result, dst_.dtype));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, data_;
};

} // namespace

InstructionPtr decodeReciprocal(Fields &fields) {
  return std::make_unique<Reciprocal>(fields);
}

} // namespace nki
