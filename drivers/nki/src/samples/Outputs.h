// The outputs' files: <dir>/<name>.bin holds the output of that name.
#pragma once

#include <string>
#include <vector>

#include "accelerator/Accelerator.h"
#include "trace/Kernel.h"

namespace nki {

// Writes the outputs, each to <dir>/<name>.bin.
void storeOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  const std::string &dir);

} // namespace nki
