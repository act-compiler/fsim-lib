// The outputs' files: <dir>/<name>.bin holds every simulated sample's output, in sample order.
// Before the first byte, every sample's outputs must fit 1 GiB and half the free disk.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accelerator/Accelerator.h"
#include "trace/Kernel.h"

namespace nki {

// Refuses numSamples samples' outputs past the cap or half of dir's free disk.
void checkOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  size_t numSamples, const std::string &dir);
// Writes batchIndex's outputs as sample number sample of each <dir>/<name>.bin, in order from 0.
void storeOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  uint32_t batchIndex, size_t sample, const std::string &dir);

} // namespace nki
