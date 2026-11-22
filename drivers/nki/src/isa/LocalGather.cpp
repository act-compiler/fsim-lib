// local_gather: each 16-partition GpSimd core gathers src columns by its indices.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

constexpr int64_t kCore = 16;

class LocalGather final : public Instruction {
public:
  explicit LocalGather(Fields &f)
      : dst_(f.view("dst")), src_(f.view("src_buffer")), index_(f.view("index")) {
    Number per = f.number("num_elem_per_idx");
    if (!per.integer || *per.integer != 1)
      f.fail("num_elem_per_idx", "other than 1 is not decoded");
    Number valid = f.number("num_valid_indices");
    if (!valid.integer || *valid.integer != dst_.perPartition())
      f.fail("num_valid_indices", "other than dst's elements per partition is not decoded");
    if (index_.dtype != Dtype::u16)
      f.fail("index", "is not u16");
    if (rows() % kCore || src_.partitions().count != rows() || index_.partitions().count != rows())
      f.fail("dst", "partitions are not whole GpSimd cores alike in src_buffer and index");
    if (dst_.perPartition() > index_.perPartition() * kCore)
      f.fail("num_valid_indices", "exceeds a GpSimd core's indices");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int64_t cores = rows() / kCore, n = dst_.perPartition(), columns = index_.perPartition();
    int64_t srcFree = src_.perPartition();
    // Each GpSimd core's indices, flattened partition-first.
    int idx = module.convert(module.read(index_), "i32");
    idx = module.reshape(idx, {cores, kCore, columns});
    idx = module.reshape(module.transpose(idx, {0, 2, 1}), {cores, columns * kCore});
    idx = module.slice(idx, {0, 0}, {cores, n}, {1, 1});
    // Clipped to the row, as nki.simulate does.
    idx = module.binary("minimum", idx,
                        module.fill(Dtype::i32, {double(srcFree - 1), srcFree - 1}, {cores, n}));
    // Spread over the core's partitions, offset by p * srcFree.
    idx = module.reshape(module.broadcast(idx, {cores, kCore, n}, {0, 2}), {rows(), n});
    std::vector<int64_t> starts;
    for (int64_t p = 0; p < rows(); ++p)
      starts.push_back(p * srcFree);
    int base = module.i32(starts);
    idx = module.binary("add", idx, module.broadcast(base, {rows(), n}, {0}));
    // One flat gather over src.
    int src = module.reshape(module.read(src_), {rows() * srcFree});
    int out = module.gather(src, module.reshape(idx, {rows() * n, 1}));
    module.write(dst_, module.reshape(out, dst_.shape()));
    accelerator.wrote(dst_);
  }

private:
  int64_t rows() const {
    return dst_.partitions().count;
  }

  const View dst_, src_, index_;
};

} // namespace

InstructionPtr decodeLocalGather(Fields &fields) {
  return std::make_unique<LocalGather>(fields);
}

} // namespace nki
