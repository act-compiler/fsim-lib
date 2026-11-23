// affine_select: on_true_tile where the affine predicate holds, else a scalar.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "isa/Shared.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

// cmp_op as a StableHLO comparison direction.
std::string direction(Fields &f) {
  std::string op = f.word("cmp_op");
  auto it = compares().find(op);
  if (it == compares().end())
    f.fail("cmp_op", "'" + op + "' is not a comparison");
  return it->second;
}

class AffineSelect final : public Instruction {
public:
  explicit AffineSelect(Fields &f)
      : dst_(f.view("dst")), pred_(f.affine("pred")), multiplier_(f.number("channel_multiplier")),
        onTrue_(f.view("on_true_tile")), onFalse_(f.number("on_false_value")),
        direction_(direction(f)) {
    if (!multiplier_.integer)
      f.fail("channel_multiplier", "is not an integer");
    if (dst_.arena != Arena::sbuf)
      f.fail("dst", "is not in sbuf");
    if (onTrue_.arena != Arena::sbuf)
      f.fail("on_true_tile", "is not in sbuf");
    if (onTrue_.partitions().count != partitions())
      f.fail("on_true_tile", "has other partitions than dst");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    // Affine value: offset + channel * multiplier + the pattern's index.
    std::vector<int64_t> rows, cols = {0};
    for (int64_t p = 0; p < partitions(); ++p)
      rows.push_back(pred_.offset + p * *multiplier_.integer);
    for (const Entry &e : pred_.pattern) {
      std::vector<int64_t> next;
      for (int64_t c : cols)
        for (int64_t i = 0; i < e.count; ++i)
          next.push_back(c + i * e.stride);
      cols = std::move(next);
    }
    std::vector<int64_t> dims = {partitions(), static_cast<int64_t>(cols.size())};
    int value = module.binary("add", module.broadcast(module.i32(rows), dims, {0}),
                              module.broadcast(module.i32(cols), dims, {1}));
    int pred = module.compare(direction_, value, module.constant({"i32", dims}, "0"));
    selectView(accelerator, module, dst_, pred, onTrue_, onFalse_.value, dims);
  }

private:
  int64_t partitions() const {
    return dst_.partitions().count;
  }

  const View dst_;
  const Affine pred_;
  const Number multiplier_;
  const View onTrue_;
  const Number onFalse_;
  const std::string direction_;
};

} // namespace

InstructionPtr decodeAffineSelect(Fields &fields) {
  return std::make_unique<AffineSelect>(fields);
}

} // namespace nki
