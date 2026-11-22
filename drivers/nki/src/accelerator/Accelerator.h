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
  // Marks PSUM bytes written by the view's allocation.
  void wrote(const View &view);
  // Per PSUM element, whether its allocation wrote it.
  std::vector<bool> started(const View &view) const;
  // Clears the mask over the view's allocation, and its clobbered entry.
  void restart(const View &view);
  // Writes an output's bytes to <dir>/<name>.bin.
  void store(const Output &output, const std::string &dir) const;
  // Refuses the view's allocation if another overwrote it.
  void unclobbered(const View &view) const;

  std::vector<uint8_t> hbm, sbuf, psum;
  std::vector<float> reduceRegs; // f32[128, 1]
  std::map<int64_t, Alloc> allocs;
  // A span of PSUM bytes to end, last written by alloc.
  struct Owner {
    int64_t end;
    int64_t alloc;
  };

private:
  // Refuses a view off its allocation, arena or dtype grid.
  void check(const View &view) const;

  // Per PSUM partition: the written spans by first byte; unwritten bytes have none.
  std::vector<std::map<int64_t, Owner>> written;
  std::map<int64_t, int64_t> clobbered; // allocation -> the last that overwrote its bytes
};

} // namespace nki
