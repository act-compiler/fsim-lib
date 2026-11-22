// The Accelerator: the arenas' host copies, the reduce and accumulator registers,
// the allocation table and the PSUM written mask.
// Host copies are the pipeline's state after a sync, each holding the batch's samples back to back.
// Host reads during a trace address batch index 0, the first.
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

  // Sizes each host copy to batchSize samples back to back, zeroed.
  void hold(uint32_t batchSize);
  // Bytes per partition row; hbm is one row.
  int64_t rowBytes(Arena arena) const;
  // Partitions of the arena.
  int64_t partitions(Arena arena) const;
  // The host byte at a partition and byte, at batch index 0.
  uint8_t *at(Arena arena, int64_t partition, int64_t byte);
  // The host byte at a partition and byte, in batchIndex's part of the host copy.
  uint8_t *at(Arena arena, int64_t partition, int64_t byte, uint32_t batchIndex);
  const uint8_t *at(Arena arena, int64_t partition, int64_t byte, uint32_t batchIndex) const;

  // Records an allocation.
  void allocate(const Alloc &alloc);
  // Forgets the allocations, the written mask and clobbers.
  void reset();
  // Zeroes batchIndex's part of each host copy.
  void zero(uint32_t batchIndex);
  // The view, its offset read from the host copies, checked.
  View resolve(const View &view);
  // Marks PSUM bytes written by the view's allocation.
  void wrote(const View &view);
  // Per PSUM element, whether its allocation wrote it.
  std::vector<bool> started(const View &view) const;
  // Clears the mask over the view's allocation, and its clobbered entry.
  void restart(const View &view);
  // Refuses the view's allocation if another overwrote it.
  void unclobbered(const View &view) const;

  std::vector<uint8_t> hbm, sbuf, psum;
  std::vector<float> reduceRegs;      // f32[128, 1] per sample
  std::vector<float> accumulatorRegs; // f32[128, 1] per sample, the Vector Engine's
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

  int64_t hbmRow_; // one sample's hbm bytes
};

} // namespace nki
