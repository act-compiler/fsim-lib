// The inputs: each input allocation's file holds its bytes.
#pragma once

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
  // Loads each input allocation's file into the host copies.
  void load(Accelerator &accelerator) const;

private:
  std::string dir_;
  std::vector<Alloc> allocs_;
};

} // namespace nki
