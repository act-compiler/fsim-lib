// The lowering helpers the kinds share.
#pragma once

#include "trace/Fields.h"

namespace nki {

class Accelerator;
class Module;

// Copies src to dst: bits, or through f32.
void copyView(Accelerator &accelerator, Module &module, const View &dst, const View &src);

} // namespace nki
