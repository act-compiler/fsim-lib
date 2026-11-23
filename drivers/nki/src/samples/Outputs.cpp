#include "samples/Outputs.h"

#include <fstream>
#include <stdexcept>

namespace nki {

namespace {

// The output's allocation; refuses a missing one.
const Alloc &allocation(const Accelerator &accelerator, const Output &o) {
  auto it = accelerator.allocs.find(o.id);
  if (it == accelerator.allocs.end())
    throw std::runtime_error("output " + o.name + " names no allocation %" + std::to_string(o.id));
  return it->second;
}

} // namespace

void storeOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  uint32_t batchIndex, size_t sample, const std::string &dir) {
  for (const Output &o : outputs) {
    const Alloc &a = allocation(accelerator, o);
    std::string path = dir + "/" + o.name + ".bin";
    // Sample 0 starts the file; each later sample appends.
    std::ofstream out(path, std::ios::binary | (sample ? std::ios::app : std::ios::trunc));
    for (int64_t p = a.partition; p < a.partition + a.partitions(); ++p)
      out.write(reinterpret_cast<const char *>(accelerator.at(a.arena, p, a.byte, batchIndex)),
                a.rowBytes());
    if (!out)
      throw std::runtime_error("cannot write " + path);
  }
}

} // namespace nki
