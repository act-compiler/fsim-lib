// nc_matmul: stationary.T @ moving in f32, per the PSUM mask.
// accumulate=true adds; false restarts the tile, then overwrites.
// none adds where a write started the bytes, else overwrites.
#include <algorithm>
#include <stdexcept>

#include "accelerator/Accelerator.h"
#include "accelerator/Limits.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

// Refuses a perf_mode M1 does not decode.
void perfMode(Fields &f) {
  std::optional<std::string> mode = f.optWord("perf_mode");
  if (mode && *mode != "none")
    f.fail("perf_mode", "'" + *mode + "' is not decoded in M1");
}

class NcMatmul final : public Instruction {
public:
  explicit NcMatmul(Fields &f)
      : dst_(f.view("dst")), stationary_(f.view("stationary")), moving_(f.view("moving")),
        isTranspose_(f.flag("is_transpose")), accumulate_(f.optFlag("accumulate")) {
    perfMode(f);
    // The tile position does not change the result.
    f.skip("row_pos");
    f.skip("col_pos");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    const View &dst = dst_;
    checkMatmul(accelerator.allocs, dst, stationary_, moving_);
    int result = flatten(module, module.toF32(module.read(stationary_)));
    if (isTranspose_) {
      result = module.transpose(result, {1, 0});
    } else {
      int moving = flatten(module, module.toF32(module.read(moving_)));
      if (module.shape(moving)[0] != module.shape(result)[0])
        throw std::runtime_error("nc_matmul contracts " + module.type(result).str() + " with " +
                                 module.type(moving).str());
      result = module.dot(result, moving);
    }
    result = module.reshape(result, dst.shape());
    // Per element, whether it adds onto the tile.
    std::vector<bool> adds(dst.count(), accumulate_ == true);
    if (accumulate_ == false)
      accelerator.restart(dst);
    else if (!accumulate_ && !isTranspose_)
      adds = accelerator.started(dst);
    if (std::find(adds.begin(), adds.end(), true) != adds.end()) {
      int sum = module.binary("add", module.toF32(module.read(dst)), result);
      bool all = std::find(adds.begin(), adds.end(), false) == adds.end();
      result = all ? sum : module.select(module.mask(adds, dst.shape()), sum, result);
    }
    module.write(dst, module.toDtype(result, dst.dtype));
    accelerator.wrote(dst);
  }

private:
  // [K, f0, f1, ...] to [K, f0 * f1 * ...].
  static int flatten(Module &module, int value) {
    std::vector<int64_t> dims = module.shape(value);
    return module.reshape(value, {dims[0], module.type(value).count() / dims[0]});
  }

  const View dst_, stationary_, moving_;
  const bool isTranspose_;
  const std::optional<bool> accumulate_;
};

} // namespace

InstructionPtr decodeNcMatmul(Fields &fields) {
  return std::make_unique<NcMatmul>(fields);
}

} // namespace nki
