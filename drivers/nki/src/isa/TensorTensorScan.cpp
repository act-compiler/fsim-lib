// tensor_tensor_scan: r[i] = data0[i] * r[i-1] + data1[i], per partition.
// A Hillis-Steele scan composes the affine maps.
#include "accelerator/Accelerator.h"
#include "isa/Instruction.h"
#include "stablehlo/Module.h"

namespace nki {

namespace {

// x shifted right by k along the free dim, filled with fill.
int shifted(Module &module, int x, int64_t k, double fill) {
  std::vector<int64_t> dims = module.shape(x);
  int kept = module.slice(x, {0, 0}, {dims[0], dims[1] - k}, {1, 1});
  return module.update(module.f32(fill, dims), kept, {0, k});
}

class TensorTensorScan final : public Instruction {
public:
  explicit TensorTensorScan(Fields &f)
      : dst_(f.view("dst")), data0_(f.view("data0")), data1_(f.view("data1")),
        initial_(f.view("initial")) {
    f.expect("op0", "multiply");
    f.expect("op1", "add");
    f.refuse("reverse0");
    f.refuse("reverse1");
  }

  void semantics(Accelerator &accelerator, Module &module) const override {
    int64_t p = dst_.shape()[0];
    int64_t n = dst_.count() / p;
    // Each element is the map r -> a * r + b.
    int a = module.reshape(module.toF32(module.read(data0_)), {p, n});
    int b = module.reshape(module.toF32(module.read(data1_)), {p, n});
    // Compose with the map k back; the identity fills the front.
    for (int64_t k = 1; k < n; k *= 2) {
      int prevA = shifted(module, a, k, 1.0);
      int prevB = shifted(module, b, k, 0.0);
      b = module.binary("add", module.binary("multiply", a, prevB), b);
      a = module.binary("multiply", a, prevA);
    }
    // Apply each prefix map to initial.
    int init = module.reshape(module.toF32(module.read(initial_)), {p, 1});
    init = module.broadcast(init, {p, n}, {0, 1});
    int result = module.binary("add", module.binary("multiply", a, init), b);
    module.write(dst_, module.toDtype(module.reshape(result, dst_.shape()), dst_.dtype));
    accelerator.wrote(dst_);
  }

private:
  const View dst_, data0_, data1_, initial_;
};

} // namespace

InstructionPtr decodeTensorTensorScan(Fields &fields) {
  return std::make_unique<TensorTensorScan>(fields);
}

} // namespace nki
