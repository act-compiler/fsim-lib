// The outputs' files: <dir>/<name>.bin holds every simulated sample's output, in sample order.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accelerator/Accelerator.h"
#include "trace/Kernel.h"

namespace nki {

// Writes batchIndex's outputs as sample number sample of each <dir>/<name>.bin, in order from 0.
void storeOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  uint32_t batchIndex, size_t sample, const std::string &dir);

} // namespace nki
