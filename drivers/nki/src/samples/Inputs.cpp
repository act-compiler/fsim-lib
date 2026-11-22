#include "samples/Inputs.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace nki {

Inputs::Inputs(std::string dir) : dir_(std::move(dir)) {
}

void Inputs::add(const Alloc &a) {
  // An input file holds the allocation's bytes.
  if (a.arena != Arena::hbm)
    throw std::runtime_error("input " + a.input + " is not in hbm");
  std::string path = dir_ + "/" + a.input + ".bin";
  std::error_code error;
  uintmax_t size = std::filesystem::file_size(path, error);
  if (error)
    throw std::runtime_error("cannot read " + path);
  uintmax_t one = static_cast<uintmax_t>(a.rowBytes());
  if (size != one)
    throw std::runtime_error(path + " holds " + std::to_string(size) + " bytes, input " + a.input +
                             " " + std::to_string(one));
  allocs_.push_back(a);
}

void Inputs::load(Accelerator &accelerator) const {
  for (const Alloc &a : allocs_) {
    std::string path = dir_ + "/" + a.input + ".bin";
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char *>(accelerator.at(Arena::hbm, 0, a.byte)), a.rowBytes());
    if (!in)
      throw std::runtime_error("cannot read " + path);
  }
}

} // namespace nki
