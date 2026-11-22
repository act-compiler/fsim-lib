// select_reduce: on_true where the predicate is nonzero, else a scalar.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class SelectReduce final : public Instruction {
public:
  explicit SelectReduce(Fields &f)
      : dst_(f.view("dst")), predicate_(f.view("predicate")), onTrue_(f.view("on_true")),
        onFalse_(f.number("on_false")) {
    // Idle, no reduce_res and no reverse are decoded.
    f.expectNone("reduce_res");
    f.expect("reduce_cmd", "idle");
    f.expect("reduce_op", "maximum");
    f.refuse("reverse_pred");
    // The docs allow an i8, u8, i16 or u16 predicate.
    if (predicate_.dtype != Dtype::u8 && predicate_.dtype != Dtype::u16)
      f.fail("predicate", "is not a u8 or u16 view");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    std::vector<int64_t> dims = dst_.shape();
    // A nonzero predicate selects, as numpy's astype(bool).
    int pred = module.reshape(module.toF32(module.read(predicate_)), dims);
    pred = module.compare("NE", pred, module.f32(0, dims));
    // Idle leaves the registers, as nki.simulate does.
    selectView(accelerator, module, dst_, pred, onTrue_, onFalse_.value, dims);
  }

private:
  const View dst_, predicate_, onTrue_;
  const Number onFalse_;
};

} // namespace

InstructionPtr decodeSelectReduce(Fields &fields) {
  return std::make_unique<SelectReduce>(fields);
}

} // namespace nki
