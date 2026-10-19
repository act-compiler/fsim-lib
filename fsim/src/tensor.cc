#include "fsim/tensor.h"

#include "pjrt.h"

namespace fsim {
namespace {

// Bytes of one element.
size_t element_size(ElementType type) {
  switch (type) {
  case ElementType::ui8:
  case ElementType::si8:
  case ElementType::f8e4m3fn:
  case ElementType::f8e5m2:
    return 1;
  case ElementType::ui16:
  case ElementType::si16:
  case ElementType::f16:
  case ElementType::bf16:
    return 2;
  case ElementType::ui32:
  case ElementType::si32:
  case ElementType::f32:
    return 4;
  case ElementType::ui64:
  case ElementType::si64:
  case ElementType::f64:
    return 8;
  }
  __builtin_unreachable();
}

} // namespace

size_t byte_size(const Tensor &tensor) {
  size_t bytes = element_size(tensor.type);
  for (int64_t dim : tensor.dims)
    bytes *= static_cast<size_t>(dim);
  return bytes;
}

PJRT_Buffer_Type buffer_type(ElementType type) {
  switch (type) {
  case ElementType::ui8:
    return PJRT_Buffer_Type_U8;
  case ElementType::ui16:
    return PJRT_Buffer_Type_U16;
  case ElementType::ui32:
    return PJRT_Buffer_Type_U32;
  case ElementType::ui64:
    return PJRT_Buffer_Type_U64;
  case ElementType::si8:
    return PJRT_Buffer_Type_S8;
  case ElementType::si16:
    return PJRT_Buffer_Type_S16;
  case ElementType::si32:
    return PJRT_Buffer_Type_S32;
  case ElementType::si64:
    return PJRT_Buffer_Type_S64;
  case ElementType::f16:
    return PJRT_Buffer_Type_F16;
  case ElementType::bf16:
    return PJRT_Buffer_Type_BF16;
  case ElementType::f32:
    return PJRT_Buffer_Type_F32;
  case ElementType::f64:
    return PJRT_Buffer_Type_F64;
  case ElementType::f8e4m3fn:
    return PJRT_Buffer_Type_F8E4M3FN;
  case ElementType::f8e5m2:
    return PJRT_Buffer_Type_F8E5M2;
  }
  __builtin_unreachable();
}

} // namespace fsim
