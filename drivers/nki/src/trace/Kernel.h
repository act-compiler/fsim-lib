// The kernel a trace records; it grows as records arrive.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "trace/View.h"

namespace nki {

// The header: the format version and the HBM arena's capacity.
struct Header {
  int64_t version = 0;
  int64_t hbm = 0;
};

// One allocation: a tensor at an address of its arena.
struct Alloc {
  int64_t id = 0;
  Arena arena = Arena::hbm;
  int64_t partition = 0;
  int64_t byte = 0;
  Dtype dtype = Dtype::u8;
  std::vector<int64_t> dims;
  bool kernel = false; // the kernel placed it
  std::string name;
  std::string input;

  // Bytes per partition, or in all for hbm.
  int64_t rowBytes() const;
  // Partitions it spans; hbm has one.
  int64_t partitions() const;
};

// A returned tensor.
struct Output {
  std::string name;
  int64_t id = 0;
};

struct Kernel {
  Header header;
  std::vector<Output> outputs;
};

} // namespace nki
