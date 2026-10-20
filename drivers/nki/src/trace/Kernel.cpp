#include "trace/Kernel.h"

namespace nki {

int64_t Alloc::rowBytes() const {
  int64_t n = width(dtype);
  for (size_t d = arena == Arena::hbm ? 0 : 1; d < dims.size(); ++d)
    n *= dims[d];
  return n;
}

int64_t Alloc::partitions() const {
  return arena == Arena::hbm || dims.empty() ? 1 : dims.front();
}

} // namespace nki
