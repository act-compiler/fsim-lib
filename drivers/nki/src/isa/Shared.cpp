#include "isa/Shared.h"

#include <set>

#include "accelerator/Accelerator.h"
#include "stablehlo/Module.h"

namespace nki {

void copyView(Accelerator &accelerator, Module &module, const View &dst, const View &src) {
  int value = module.read(src);
  if (src.dtype != dst.dtype)
    value = module.toDtype(module.toF32(value), dst.dtype);
  module.write(dst, value);
  accelerator.wrote(dst);
}

void selectView(Accelerator &accelerator, Module &module, const View &dst, int pred,
                const View &onTrue, double onFalse, const std::vector<int64_t> &dims) {
  // Selects in f32, as nki.simulate does.
  int value = module.reshape(module.toF32(module.read(onTrue)), dims);
  int result = module.select(pred, value, module.f32(onFalse, dims));
  module.write(dst, module.toDtype(module.reshape(result, dst.shape()), dst.dtype));
  accelerator.wrote(dst);
}

int applyOperand(Module &module, const std::string &op, int x, const Value &operand) {
  std::vector<int64_t> dims = module.shape(x);
  if (auto *n = std::get_if<Number>(&operand))
    return module.binary(op, x, module.f32(n->value, dims));
  int v = module.toF32(module.read(std::get<View>(operand)));
  std::vector<int64_t> padded = module.shape(v);
  if (padded.size() < dims.size())
    padded.resize(dims.size(), 1);
  auto [lhs, rhs] = module.broadcastPair(x, module.reshape(v, padded));
  return module.binary(op, lhs, rhs);
}

const std::map<std::string, std::string> &compares() {
  static const std::map<std::string, std::string> directions = {
      {"equal", "EQ"},         {"not_equal", "NE"}, {"greater", "GT"},
      {"greater_equal", "GE"}, {"less", "LT"},      {"less_equal", "LE"}};
  return directions;
}

std::string scalarOp(Fields &fields) {
  static const std::set<std::string> ops = {"add", "multiply", "maximum"};
  std::string name = fields.word("op0");
  if (!ops.count(name))
    fields.fail("op0", "'" + name + "' is not decoded");
  return name;
}

int scalarArith(Module &module, const View &dst, const View &data, const std::string &op,
                const Value &operand) {
  int x = module.toF32(module.read(data));
  if (data.count() == dst.count())
    x = module.reshape(x, dst.shape());
  return module.reshape(applyOperand(module, op, x, operand), dst.shape());
}

} // namespace nki
