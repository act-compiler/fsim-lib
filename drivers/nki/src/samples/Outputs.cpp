#include "samples/Outputs.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace nki {

namespace {

// Every simulated sample's outputs together, 1 GiB at most.
constexpr uintmax_t kOutputCap = uintmax_t{1} << 30;

// The output's allocation; refuses a missing one.
const Alloc &allocation(const Accelerator &accelerator, const Output &o) {
  auto it = accelerator.allocs.find(o.id);
  if (it == accelerator.allocs.end())
    throw std::runtime_error("output " + o.name + " names no allocation %" + std::to_string(o.id));
  return it->second;
}

} // namespace

void checkOutputs(const Accelerator &accelerator, const std::vector<Output> &outputs,
                  size_t numSamples, const std::string &dir) {
  uintmax_t total = 0;
  for (const Output &o : outputs) {
    if (o.name == "draws")
      throw std::runtime_error("output draws takes the name of " + dir + "/draws.bin");
    const Alloc &a = allocation(accelerator, o);
    total += numSamples * static_cast<uintmax_t>(a.partitions() * a.rowBytes());
  }
  uintmax_t room = std::filesystem::space(dir).available;
  if (total > room / 2 || total > kOutputCap)
    throw std::runtime_error(std::to_string(numSamples) + " samples' outputs take " +
                             std::to_string(total) + " bytes, past half the " +
                             std::to_string(room) + " bytes free on " + dir + " or the " +
                             std::to_string(kOutputCap) + "-byte cap");
}

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

void storeDraws(const std::vector<uint32_t> &draws, const std::string &dir) {
  // The host is little-endian, as setup.sh's x86_64 is.
  std::string path = dir + "/draws.bin";
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(draws.data()),
            static_cast<std::streamsize>(draws.size() * sizeof(uint32_t)));
  if (!out)
    throw std::runtime_error("cannot write " + path);
}

} // namespace nki
