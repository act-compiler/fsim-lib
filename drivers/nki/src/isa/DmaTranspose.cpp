// dma_transpose: src permuted by axes, a bit copy.
#include <algorithm>
#include <vector>

#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

// The documented permutations, one per rank.
std::vector<int64_t> axes(Fields &f, const View &src) {
  static const std::vector<std::vector<int64_t>> documented = {{1, 0}, {2, 1, 0}, {3, 1, 2, 0}};
  std::vector<int64_t> perm;
  for (const Number &n : f.list("axes"))
    perm.push_back(n.integer.value());
  if (std::find(documented.begin(), documented.end(), perm) == documented.end() ||
      perm.size() != src.pattern.size())
    f.fail("axes", "is not the documented permutation of a rank " +
                       std::to_string(src.pattern.size()) + " src");
  return perm;
}

class DmaTranspose final : public Instruction {
public:
  explicit DmaTranspose(Fields &f)
      : dst_(f.view("dst")), src_(f.view("src")), axes_(axes(f, src_)) {
    if (dst_.arena != Arena::sbuf)
      f.fail("dst", "is not in sbuf");
    if (src_.arena == Arena::psum)
      f.fail("src", "is not in hbm or sbuf");
    if (src_.dtype != dst_.dtype || width(src_.dtype) != 2)
      f.fail("src", "is not a 2-byte dtype equal to dst's");
    // Checked views make out of bounds always fail.
    f.skip("oob_is_err");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int value = module.transpose(module.read(src_), axes_);
    module.write(dst_, module.reshape(value, dst_.shape()));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, src_;
  const std::vector<int64_t> axes_;
};

} // namespace

InstructionPtr decodeDmaTranspose(Fields &fields) {
  return std::make_unique<DmaTranspose>(fields);
}

} // namespace nki
