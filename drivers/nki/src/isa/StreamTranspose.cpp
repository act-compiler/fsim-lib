// stream_transpose: data's partitions and free elements swapped.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

class StreamTranspose final : public Instruction {
public:
  explicit StreamTranspose(Fields &f) : dst_(f.view("dst")), data_(f.view("data")) {
    std::string engine = f.word("engine");
    if (engine != "vector")
      f.fail("engine", "'" + engine + "' is not the Vector Engine");
    // nki.isa.nc_transpose's Vector Engine bound.
    int64_t rows = data_.partitions().count;
    if (rows > 32 || data_.perPartition() > 32)
      f.fail("data", "is past the Vector Engine's 32x32");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    // [P, free] to [free, P], as nki.simulate's .T.
    int x = module.reshape(module.read(data_), {data_.partitions().count, data_.perPartition()});
    int out = module.transpose(x, {1, 0});
    module.write(dst_, module.reshape(out, dst_.shape()));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, data_;
};

} // namespace

InstructionPtr decodeStreamTranspose(Fields &fields) {
  return std::make_unique<StreamTranspose>(fields);
}

} // namespace nki
