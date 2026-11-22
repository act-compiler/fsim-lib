#include "isa/Shared.h"

#include "accelerator/Accelerator.h"
#include "stablehlo/Module.h"

namespace nki {

void copyView(Accelerator &accelerator, Module &module, const View &dst, const View &src) {
  int value = module.read(src);
  if (src.dtype != dst.dtype)
    value = module.toDtype(module.toF32(value), dst.dtype);
  module.write(dst, value);
  accelerator.wrote(dst);
}

void selectView(Accelerator &accelerator, Module &module, const View &dst, int pred,
                const View &onTrue, double onFalse, const std::vector<int64_t> &dims) {
  // Selects in f32, as nki.simulate does.
  int value = module.reshape(module.toF32(module.read(onTrue)), dims);
  int result = module.select(pred, value, module.f32(onFalse, dims));
  module.write(dst, module.toDtype(module.reshape(result, dst.shape()), dst.dtype));
  accelerator.wrote(dst);
}

const std::map<std::string, std::string> &compares() {
  static const std::map<std::string, std::string> directions = {
      {"equal", "EQ"},         {"not_equal", "NE"}, {"greater", "GT"},
      {"greater_equal", "GE"}, {"less", "LT"},      {"less_equal", "LE"}};
  return directions;
}

} // namespace nki
