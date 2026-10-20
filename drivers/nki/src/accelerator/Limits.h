// The memory-model checks: allocations within their arenas,
// nc_matmul's operand arenas, and its dst in one bank.
#pragma once

#include <cstdint>
#include <map>

#include "trace/Kernel.h"

namespace nki {

constexpr int64_t kPsumBank = 2048; // bytes per partition per bank

// The newest allocation holding the view's first byte, or none.
const Alloc *holder(const std::map<int64_t, Alloc> &allocs, const View &view);
// Refuses a duplicate id, bad dim or overflowing tensor.
void checkAlloc(const std::map<int64_t, Alloc> &allocs, const Alloc &alloc, int64_t hbm);
// Refuses operands off their arenas or a dst across banks.
void checkMatmul(const std::map<int64_t, Alloc> &allocs, const View &dst, const View &stationary,
                 const View &moving);

} // namespace nki
