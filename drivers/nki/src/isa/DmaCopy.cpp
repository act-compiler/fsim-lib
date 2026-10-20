// dma_copy: a bit copy, or an f32 convert.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class DmaCopy final : public Instruction {
public:
  explicit DmaCopy(Fields &f) : dst_(f.view("dst")), src_(f.view("src")) {
    // Checked views make out of bounds always fail.
    f.skip("oob_is_err");
    f.skip("engine");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int value = module.read(src_);
    if (src_.dtype != dst_.dtype)
      value = module.toDtype(module.toF32(value), dst_.dtype);
    module.write(dst_, value);
    accelerator.wrote(dst_);
  }

private:
  const View dst_, src_;
};

} // namespace

InstructionPtr decodeDmaCopy(Fields &fields) {
  return std::make_unique<DmaCopy>(fields);
}

} // namespace nki
