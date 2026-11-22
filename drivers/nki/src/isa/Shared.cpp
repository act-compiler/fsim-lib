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

} // namespace nki
