// tensor_copy: a bit copy, or an f32 convert.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class TensorCopy final : public Instruction {
public:
  explicit TensorCopy(Fields &f) : dst_(f.view("dst")), src_(f.view("src")) {
    // Every engine copies alike: bits, or through f32.
    f.skip("engine");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    copyView(accelerator, module, dst_, src_);
  }

private:
  const View dst_, src_;
};

} // namespace

InstructionPtr decodeTensorCopy(Fields &fields) {
  return std::make_unique<TensorCopy>(fields);
}

} // namespace nki
