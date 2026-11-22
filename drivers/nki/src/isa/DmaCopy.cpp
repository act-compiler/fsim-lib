// dma_copy: a bit copy, or an f32 convert.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class DmaCopy final : public Instruction {
public:
  explicit DmaCopy(Fields &f) : dst_(f.view("dst")), src_(f.view("src")) {
    // Checked views refuse out of bounds; a +s() index clamps.
    f.skip("oob_is_err");
    f.skip("engine");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    copyView(accelerator, module, dst_, src_);
  }

private:
  const View dst_, src_;
};

} // namespace

InstructionPtr decodeDmaCopy(Fields &fields) {
  return std::make_unique<DmaCopy>(fields);
}

} // namespace nki
