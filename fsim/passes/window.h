// The bytes a slice reads or a dynamic_update_slice writes of its buffer.
// A window is a strided box of elements over a tensor type; its bytes are
// sorted, disjoint, coalesced [begin, end) intervals over the flat buffer,
// which reshapes and equal-width bitcasts keep in place.
#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

namespace fsim {

// Flat byte intervals, sorted, disjoint and coalesced.
using Intervals = std::vector<std::pair<int64_t, int64_t>>;

// How window a relates to window b.
enum class Relation { Disjoint, Equal, Inside, Other };

class Window {
public:
  // The slice's window over its operand.
  static std::optional<Window> of(mlir::stablehlo::SliceOp slice);
  // The update's window, start indices clamped; none unless constant.
  static std::optional<Window> of(mlir::stablehlo::DynamicUpdateSliceOp update);

  // The window's bytes, or null past the interval cap.
  const Intervals *bytes() const;

  // The box: per dim, the first element, the count and the stride.
  llvm::SmallVector<int64_t> dims, start, count, stride;
  // Bytes per element.
  int64_t width = 0;

private:
  mutable bool computed_ = false;
  mutable std::optional<Intervals> bytes_;
};

// How a relates to b: boxes over one shape first, then bytes.
Relation relate(const Window &a, const Window &b);
// Whether every byte of a lies in u.
bool covered(const Intervals &a, const Intervals &u);
// Adds a's bytes to u.
void unite(Intervals &u, const Intervals &a);

// Whether op keeps its operand's bytes in order: reshape, equal-width bitcast.
bool isView(mlir::Operation *op);

} // namespace fsim
