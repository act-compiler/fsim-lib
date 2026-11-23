#include "samples/Inputs.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <utility>

namespace nki {

namespace {

// The draws' fixed seed.
constexpr uint64_t kSeed = 0;

} // namespace

Inputs::Inputs(std::string dir, const std::string &sampling)
    : dir_(std::move(dir)), modulo_(sampling == "modulo") {
  if (!modulo_ && sampling != "random")
    throw std::runtime_error("--sampling takes random or modulo, not '" + sampling + "'");
}

void Inputs::add(const Alloc &a) {
  // An input file holds its tensor of each input set back to back; every input the same count.
  if (a.arena != Arena::hbm)
    throw std::runtime_error("input " + a.input + " is not in hbm");
  std::string path = dir_ + "/" + a.input + ".bin";
  std::error_code error;
  uintmax_t size = std::filesystem::file_size(path, error);
  if (error)
    throw std::runtime_error("cannot read " + path);
  uintmax_t one = static_cast<uintmax_t>(a.rowBytes());
  if (!size || size % one)
    throw std::runtime_error(path + " holds " + std::to_string(size) +
                             " bytes, not whole tensors of " + a.input + "'s " +
                             std::to_string(one) + " bytes");
  uint32_t held = static_cast<uint32_t>(size / one);
  if (!allocs_.empty() && held != inputSets_)
    throw std::runtime_error(path + " holds " + std::to_string(held) + " tensors, but " + dir_ +
                             "/" + allocs_.front().input + ".bin " + std::to_string(inputSets_));
  inputSets_ = held;
  allocs_.push_back(a);
}

void Inputs::draw(size_t count) {
  // Either way, a longer run's draws extend a shorter one's.
  draws_.resize(count);
  if (modulo_) {
    for (size_t k = 0; k < count; ++k)
      draws_[k] = static_cast<uint32_t>(k % inputSets_);
    return;
  }
  std::mt19937_64 random(kSeed);
  std::uniform_int_distribution<uint32_t> uniform(0, inputSets_ - 1);
  for (uint32_t &draw : draws_)
    draw = uniform(random);
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
