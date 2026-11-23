// nc_stream_shuffle: each quadrant's dst partitions copy one src partition.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

constexpr int64_t kQuadrant = 32;

// The mask's one entry; only a constant mask is decoded.
int64_t constantMask(Fields &f) {
  List mask = f.list("shuffle_mask");
  if (static_cast<int64_t>(mask.size()) != kQuadrant)
    f.fail("shuffle_mask", "has other than 32 entries");
  int64_t k = mask[0].integer.value();
  for (const Number &n : mask)
    if (n.integer.value() != k)
      f.fail("shuffle_mask", "a mask of differing entries is not decoded");
  if (k < 0 || k >= kQuadrant)
    f.fail("shuffle_mask", "an entry past the quadrant, 255 included, is not decoded");
  return k;
}

class NcStreamShuffle final : public Instruction {
public:
  explicit NcStreamShuffle(Fields &f)
      : dst_(f.view("dst")), src_(f.view("src")), entry_(constantMask(f)) {
    // Every dst quadrant's source partition exists.
    if ((quadrants() - 1) * kQuadrant + entry_ >= srcRows())
      f.fail("src", "lacks a quadrant's source partition");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int64_t free = src_.perPartition();
    int src = module.reshape(module.read(src_), {srcRows(), free});
    // Each quadrant's source row, a strided slice.
    int64_t last = (quadrants() - 1) * kQuadrant + entry_;
    int rows = module.slice(src, {entry_, 0}, {last + 1, free}, {kQuadrant, 1});
    // Broadcast across the quadrant, cut to dst's partitions.
    int spread = module.broadcast(rows, {quadrants(), kQuadrant, free}, {0, 2});
    int out = module.reshape(spread, {quadrants() * kQuadrant, free});
    if (quadrants() * kQuadrant != dstRows())
      out = module.slice(out, {0, 0}, {dstRows(), free}, {1, 1});
    module.write(dst_, module.reshape(out, dst_.shape()));
    accelerator.wrote(dst_);
  }

private:
  int64_t dstRows() const {
    return dst_.partitions().count;
  }
  int64_t srcRows() const {
    return src_.partitions().count;
  }
  int64_t quadrants() const {
    return (dstRows() + kQuadrant - 1) / kQuadrant;
  }

  const View dst_, src_;
  const int64_t entry_;
};

} // namespace

InstructionPtr decodeNcStreamShuffle(Fields &fields) {
  return std::make_unique<NcStreamShuffle>(fields);
}

} // namespace nki
