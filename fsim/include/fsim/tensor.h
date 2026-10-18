// A host tensor: element type, dims, and the memory holding it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fsim {

// The element types StableHLO text declares.
enum class ElementType {
  ui8,
  ui16,
  ui32,
  ui64,
  si8,
  si16,
  si32,
  si64,
  f16,
  bf16,
  f32,
  f64,
  f8e4m3fn,
  f8e5m2,
};

// Dense row-major tensor in host memory the caller owns.
struct Tensor {
  ElementType type;
  std::vector<int64_t> dims;
  void *data;
};

// Bytes of one element.
size_t element_size(ElementType type);

// Bytes of the whole tensor.
size_t byte_size(const Tensor &tensor);

} // namespace fsim
