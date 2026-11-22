// The lowering helpers the kinds share.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "trace/Fields.h"

namespace nki {

class Accelerator;
class Module;

// Copies src to dst: bits, or through f32.
void copyView(Accelerator &accelerator, Module &module, const View &dst, const View &src);

// Writes on_true where pred holds, else on_false, in f32.
void selectView(Accelerator &accelerator, Module &module, const View &dst, int pred,
                const View &onTrue, double onFalse, const std::vector<int64_t> &dims);

// The compare ops, as StableHLO directions.
const std::map<std::string, std::string> &compares();

} // namespace nki
