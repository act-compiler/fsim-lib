#include "accelerator/Limits.h"

#include <stdexcept>
#include <string>

#include "accelerator/Accelerator.h"

namespace nki {

namespace {

[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error(what);
}

std::string str(int64_t n) {
  return std::to_string(n);
}

// Names an allocation by its placement and address.
std::string named(const Alloc &a) {
  std::string at = a.arena == Arena::hbm ? str(a.byte) : str(a.partition) + "," + str(a.byte);
  return std::string(a.kernel ? "kernel" : "driver") + " allocation %" + str(a.id) + " at " +
         name(a.arena) + "@" + at;
}

// An operand in the wrong arena.
void arena(const View &v, Arena want, const char *operand) {
  if (v.arena != want)
    fail(std::string("nc_matmul ") + operand + " " + v.address() + " is not in " + name(want) +
         "; nc_matmul " + (want == Arena::psum ? "writes" : "reads") + " it there");
}

} // namespace

const Alloc *holder(const std::map<int64_t, Alloc> &allocs, const View &v) {
  for (auto it = allocs.rbegin(); it != allocs.rend(); ++it) {
    const Alloc &a = it->second;
    if (a.arena == v.arena && v.partition >= a.partition &&
        v.partition < a.partition + a.partitions() && v.byte >= a.byte &&
        v.byte < a.byte + a.rowBytes())
      return &a;
  }
  return nullptr;
}

void checkAlloc(const std::map<int64_t, Alloc> &allocs, const Alloc &a, int64_t hbm) {
  if (allocs.count(a.id))
    fail("allocation %" + str(a.id) + " declared twice");
  for (int64_t d : a.dims)
    if (d <= 0)
      fail(named(a) + " has a dim of " + str(d));
  // SBUF and PSUM have 128 partitions.
  int64_t last = a.partition + a.partitions() - 1;
  if (a.partition < 0 || last >= kPartitions)
    fail(named(a) + " spans partitions " + str(a.partition) + " to " + str(last) + ", past the " +
         str(kPartitions) + " partitions of " + name(a.arena));
  // Bytes per partition: SBUF 229376, PSUM 16384, HBM the header's.
  int64_t row = a.arena == Arena::hbm ? hbm : a.arena == Arena::sbuf ? kSbufRow : kPsumRow;
  int64_t end = a.byte + a.rowBytes() - 1;
  std::string limit = a.arena == Arena::hbm
                          ? "the header's hbm=" + str(row)
                          : "the " + str(row) + " bytes per partition of " + name(a.arena);
  if (a.byte < 0 || end >= row)
    fail(named(a) + " spans bytes " + str(a.byte) + " to " + str(end) + ", past " + limit);
}

void checkMatmul(const std::map<int64_t, Alloc> &allocs, const View &dst, const View &stationary,
                 const View &moving) {
  arena(stationary, Arena::sbuf, "stationary");
  arena(moving, Arena::sbuf, "moving");
  arena(dst, Arena::psum, "dst");
  // A run-time offset is checked once resolved.
  if (dst.dyn)
    return;
  // The dst's bytes within a partition lie in one bank.
  int64_t k = width(dst.dtype);
  auto [lo, hi] = span(dst.byte / k, dst.free());
  int64_t first = lo * k, last = hi * k + k - 1;
  const Alloc *a = holder(allocs, dst);
  std::string in = a ? " in " + named(*a) : "";
  if (first / kPsumBank != last / kPsumBank)
    fail("nc_matmul dst " + dst.address() + in + " spans bytes " + str(first) + " to " + str(last) +
         ", across a " + str(kPsumBank) + "-byte PSUM bank boundary at " +
         str((first / kPsumBank + 1) * kPsumBank));
}

} // namespace nki
