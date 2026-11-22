// The M input sets: each input allocation's file holds its tensor of each, back to back.
// --sampling random draws each sample's input set uniformly with replacement, from a
// fixed seed, the same on every run with one C++ standard library; modulo gives sample k
// input set k mod M.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accelerator/Accelerator.h"
#include "trace/Kernel.h"

namespace nki {

class Inputs {
public:
  // dir holds each input's <name>.bin; refuses a --sampling value but random or modulo.
  Inputs(std::string dir, const std::string &sampling);

  // Adds an input allocation; refuses one off hbm, or a file other than M whole tensors.
  void add(const Alloc &alloc);
  // Draws count samples' input sets.
  void draw(size_t count);
  // The count of samples whose input set is drawn.
  size_t size() const;
  // Zeroes batchIndex's part of each host copy, then loads sample's input set into it.
  void load(Accelerator &accelerator, uint32_t batchIndex, size_t sample) const;
  // The input-set index drawn for each sample.
  const std::vector<uint32_t> &draws() const;

private:
  std::string dir_;
  bool modulo_;
  std::vector<Alloc> allocs_;
  uint32_t inputSets_ = 1; // M
  std::vector<uint32_t> draws_;
};

} // namespace nki
