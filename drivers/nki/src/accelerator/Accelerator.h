// The Accelerator: the arenas' host copies, the reduce registers,
// the allocation table and the PSUM written mask.
// Host copies are the pipeline's state after a sync.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "trace/Kernel.h"

namespace nki {

constexpr int64_t kPartitions = 128;
constexpr int64_t kSbufRow = 229376; // 224 KiB per partition
constexpr int64_t kPsumRow = 16384;  // 8 banks of 2 KiB

class Accelerator {
public:
  explicit Accelerator(int64_t hbm);

  // Bytes per partition row; hbm is one row.
  int64_t rowBytes(Arena arena) const;
  // Partitions of the arena.
  int64_t partitions(Arena arena) const;
  // The host byte at a partition and byte.
  uint8_t *at(Arena arena, int64_t partition, int64_t byte);

  // Records an allocation and loads its input.
  void allocate(const Alloc &alloc, const std::string &inputs);
  // The view, its offset read from the host copies, checked.
  View resolve(const View &view);
  // Marks PSUM bytes written, by any instruction.
  void wrote(const View &view);
  // Per PSUM element, whether written since its allocation.
  std::vector<bool> started(const View &view) const;
  // Clears the mask over the newest allocation holding it.
  void restart(const View &view);
  // Writes an output's bytes to <dir>/<name>.bin.
  void store(const Output &output, const std::string &dir) const;

  std::vector<uint8_t> hbm, sbuf, psum;
  std::vector<float> reduceRegs; // f32[128, 1]
  std::map<int64_t, Alloc> allocs;
  std::vector<uint8_t> written; // per PSUM byte: written since allocated

private:
  // Refuses a view off its arena or dtype grid.
  void check(const View &view) const;
  // Sets the mask over a view's bytes.
  void mark(const View &view, uint8_t value);
  // Clears the mask over an allocation's bytes.
  void unwrite(const Alloc &alloc);
};

} // namespace nki
