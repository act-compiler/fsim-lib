// memset: the value in the destination's dtype, everywhere.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class Memset final : public Instruction {
public:
  explicit Memset(Fields &f) : dst_(f.view("dst")), value_(f.number("value")) {
    f.skip("engine");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    module.write(dst_, module.fill(dst_.dtype, value_, dst_.shape()));
    accelerator.wrote(dst_);
  }

private:
  const View dst_;
  const Number value_;
};

} // namespace

InstructionPtr decodeMemset(Fields &fields) {
  return std::make_unique<Memset>(fields);
}

} // namespace nki
