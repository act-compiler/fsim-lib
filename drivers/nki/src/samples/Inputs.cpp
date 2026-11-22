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

void Inputs::draw(size_t count) {
  draws_.resize(count);
}

size_t Inputs::size() const {
  return draws_.size();
}

void Inputs::load(Accelerator &accelerator, uint32_t batchIndex, size_t sample) const {
  accelerator.zero(batchIndex);
  // Each input's tensor of input set s sits at s times its bytes.
  uint32_t s = draws_[sample];
  for (const Alloc &a : allocs_) {
    std::string path = dir_ + "/" + a.input + ".bin";
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(s) * a.rowBytes());
    in.read(reinterpret_cast<char *>(accelerator.at(Arena::hbm, 0, a.byte, batchIndex)),
            a.rowBytes());
    if (!in)
      throw std::runtime_error("cannot read input set " + std::to_string(s) + "'s tensor from " +
                               path);
  }
}

const std::vector<uint32_t> &Inputs::draws() const {
  return draws_;
}

} // namespace nki
