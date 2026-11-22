// The inputs: each input allocation's file holds its bytes.
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
  // dir holds each input's <name>.bin.
  explicit Inputs(std::string dir);

  // Adds an input allocation; refuses one off hbm, or a file of other than its bytes.
  void add(const Alloc &alloc);
  // Draws count samples' input sets.
  void draw(size_t count);
  // The count of samples whose input set is drawn.
  size_t size() const;
  // Zeroes batchIndex's part of each host copy, then reads each input's tensor.
  void load(Accelerator &accelerator, uint32_t batchIndex, size_t sample) const;
  // The input-set index drawn for each sample.
  const std::vector<uint32_t> &draws() const;

private:
  std::string dir_;
  std::vector<Alloc> allocs_;
  std::vector<uint32_t> draws_;
};

} // namespace nki
