#include "accelerator/Accelerator.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "accelerator/Limits.h"

namespace nki {

namespace {

[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error(what);
}

// The header's capacity: positive, a multiple of 4.
int64_t capacity(int64_t hbm) {
  if (hbm <= 0 || hbm % 4 != 0)
    fail("hbm=" + std::to_string(hbm) + " is not a positive multiple of 4 bytes");
  return hbm;
}

// Calls f(partition, element, n) per span of n adjacent elements.
template <class F> void spans(const View &v, F f) {
  if (v.pattern.size() < 2 || v.pattern.back().stride != 1) {
    v.each([&](int64_t p, int64_t e) { f(p, e, 1); });
    return;
  }
  View rows = v;
  rows.pattern.back().count = 1;
  int64_t n = v.pattern.back().count;
  rows.each([&](int64_t p, int64_t e) { f(p, e, n); });
}

// Splits the span across byte at, so one starts there.
void cut(std::map<int64_t, Accelerator::Owner> &row, int64_t at) {
  auto it = row.upper_bound(at);
  if (it == row.begin())
    return;
  --it;
  if (it->first < at && it->second.end > at) {
    row.emplace(at, it->second);
    it->second.end = at;
  }
}

} // namespace

Accelerator::Accelerator(int64_t hbm) : written(kPartitions), hbmRow_(capacity(hbm)) {
}

void Accelerator::hold(uint32_t batchSize) {
  hbm.assign(batchSize * hbmRow_, 0);
  sbuf.assign(batchSize * kPartitions * kSbufRow, 0);
  psum.assign(batchSize * kPartitions * kPsumRow, 0);
  reduceRegs.assign(batchSize * kPartitions, 0.0f);
  accumulatorRegs.assign(batchSize * kPartitions, 0.0f);
}

int64_t Accelerator::rowBytes(Arena arena) const {
  switch (arena) {
  case Arena::hbm:
    return hbmRow_;
  case Arena::sbuf:
    return kSbufRow;
  case Arena::psum:
    return kPsumRow;
  }
  return 0;
}

int64_t Accelerator::partitions(Arena arena) const {
  return arena == Arena::hbm ? 1 : kPartitions;
}

uint8_t *Accelerator::at(Arena arena, int64_t partition, int64_t byte) {
  std::vector<uint8_t> &bytes = arena == Arena::hbm ? hbm : arena == Arena::sbuf ? sbuf : psum;
  return bytes.data() + partition * rowBytes(arena) + byte;
}

uint8_t *Accelerator::at(Arena arena, int64_t partition, int64_t byte, uint32_t batchIndex) {
  // Batch index m's partition p is the host row m * partitions + p.
  return at(arena, batchIndex * partitions(arena) + partition, byte);
}

const uint8_t *Accelerator::at(Arena arena, int64_t partition, int64_t byte,
                               uint32_t batchIndex) const {
  return const_cast<Accelerator *>(this)->at(arena, partition, byte, batchIndex);
}

void Accelerator::allocate(const Alloc &a) {
  checkAlloc(allocs, a, hbmRow_);
  allocs.emplace(a.id, a);
}

void Accelerator::zero(uint32_t batchIndex) {
  auto zero = [&](auto &array, int64_t size) {
    std::fill_n(array.begin() + batchIndex * size, size, 0);
  };
  zero(hbm, hbmRow_);
  zero(sbuf, kPartitions * kSbufRow);
  zero(psum, kPartitions * kPsumRow);
  zero(reduceRegs, kPartitions);
  zero(accumulatorRegs, kPartitions);
}

void Accelerator::check(const View &v) const {
  std::string where = v.address();
  auto it = allocs.find(v.alloc);
  if (it == allocs.end() || it->second.arena != v.arena)
    fail("view " + where + " names no " + name(v.arena) + " allocation");
  int64_t k = width(v.dtype);
  for (const Entry &e : v.pattern)
    if (e.count <= 0)
      fail("view " + where + " has a count of " + std::to_string(e.count));
  if (v.byte % k != 0)
    fail("view " + where + " is not aligned to its dtype's " + std::to_string(k) + " bytes");
  Entry p = v.partitions();
  auto [plo, phi] = span(v.partition, {p});
  if (plo < 0 || phi >= partitions(v.arena))
    fail("view " + where + " crosses its arena's partitions");
  auto [lo, hi] = span(v.byte / k, v.free());
  if (lo < 0 || hi >= rowBytes(v.arena) / k)
    fail("view " + where + " crosses its arena's partition row");
  // The view stays inside its allocation.
  const Alloc &a = it->second;
  if (plo < a.partition || phi >= a.partition + a.partitions())
    fail("view " + where + " leaves allocation %" + std::to_string(a.id) + "'s partitions " +
         std::to_string(a.partition) + " to " + std::to_string(a.partition + a.partitions() - 1));
  if (lo * k < a.byte || hi * k + k > a.byte + a.rowBytes())
    fail("view " + where + " leaves allocation %" + std::to_string(a.id) + "'s bytes " +
         std::to_string(a.byte) + " to " + std::to_string(a.byte + a.rowBytes() - 1));
}

View Accelerator::resolve(const View &view) {
  View v = view;
  if (v.dyn) {
    View inner = resolve(*v.dyn->view);
    if ((inner.dtype != Dtype::i32 && inner.dtype != Dtype::u32) || inner.count() != 1)
      fail("a +s() offset reads one i32 or u32; a vector offset is not decoded");
    int32_t offset;
    std::memcpy(&offset, at(inner.arena, inner.partition, inner.byte), sizeof offset);
    v.byte += offset * v.dyn->scale * width(v.dtype);
    v.dyn.reset();
  }
  check(v);
  return v;
}

void Accelerator::wrote(const View &v) {
  if (v.arena != Arena::psum)
    return;
  int64_t k = width(v.dtype);
  spans(v, [&](int64_t p, int64_t e, int64_t n) {
    std::map<int64_t, Owner> &row = written[p];
    int64_t lo = e * k, hi = (e + n) * k;
    cut(row, lo);
    cut(row, hi);
    auto first = row.lower_bound(lo), last = row.lower_bound(hi);
    // Taking another allocation's written bytes clobbers it.
    for (auto it = first; it != last; ++it)
      if (it->second.alloc != v.alloc)
        clobbered[it->second.alloc] = v.alloc;
    row.erase(first, last);
    row.emplace(lo, Owner{hi, v.alloc});
  });
}

std::vector<bool> Accelerator::started(const View &v) const {
  std::vector<bool> out;
  out.reserve(v.count());
  int64_t k = width(v.dtype);
  spans(v, [&](int64_t p, int64_t e, int64_t n) {
    const std::map<int64_t, Owner> &row = written[p];
    int64_t x = e * k, hi = (e + n) * k;
    auto it = row.upper_bound(x);
    if (it != row.begin())
      --it;
    while (x < hi) {
      // The next span of the allocation's bytes past x.
      while (it != row.end() && (it->second.end <= x || it->second.alloc != v.alloc))
        ++it;
      // Elements before its first byte hold another's byte.
      int64_t from = it == row.end() ? hi : std::min(std::max(it->first, x), hi);
      int64_t before = (from - x + k - 1) / k;
      out.insert(out.end(), before, false);
      x += before * k;
      if (x >= hi || x >= it->second.end)
        continue;
      // Adjacent spans of the allocation extend this one to end.
      int64_t end = it->second.end;
      for (auto next = std::next(it);
           next != row.end() && next->first == end && next->second.alloc == v.alloc; ++next)
        end = next->second.end;
      int64_t inside = (std::min(end, hi) - x) / k;
      // An element across that end is not started.
      out.insert(out.end(), std::max<int64_t>(inside, 1), inside > 0);
      x += std::max<int64_t>(inside, 1) * k;
    }
  });
  return out;
}

void Accelerator::restart(const View &v) {
  if (v.arena != Arena::psum)
    return;
  // Another allocation's bytes keep their writer.
  const Alloc &a = allocs.at(v.alloc);
  for (int64_t p = a.partition; p < a.partition + a.partitions(); ++p) {
    std::map<int64_t, Owner> &row = written[p];
    int64_t lo = a.byte, hi = a.byte + a.rowBytes();
    cut(row, lo);
    cut(row, hi);
    for (auto it = row.lower_bound(lo); it != row.end() && it->first < hi;)
      it = it->second.alloc == v.alloc ? row.erase(it) : std::next(it);
  }
  clobbered.erase(v.alloc);
}

void Accelerator::unclobbered(const View &v) const {
  auto it = clobbered.find(v.alloc);
  if (it != clobbered.end())
    fail("psum allocation %" + std::to_string(v.alloc) + " accumulates after allocation %" +
         std::to_string(it->second) + " overwrote its bytes; aliased accumulation is refused");
}

} // namespace nki
