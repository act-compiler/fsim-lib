#include "accelerator/Accelerator.h"

#include <algorithm>
#include <cstring>
#include <fstream>
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

} // namespace

Accelerator::Accelerator(int64_t hbm)
    : hbm(capacity(hbm)), sbuf(kPartitions * kSbufRow), psum(kPartitions * kPsumRow),
      reduceRegs(kPartitions), written(kPartitions * kPsumRow) {
}

int64_t Accelerator::rowBytes(Arena arena) const {
  switch (arena) {
  case Arena::hbm:
    return static_cast<int64_t>(hbm.size());
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

void Accelerator::allocate(const Alloc &a, const std::string &inputs) {
  checkAlloc(allocs, a, static_cast<int64_t>(hbm.size()));
  allocs.emplace(a.id, a);
  // A new PSUM tensor starts unwritten.
  if (a.arena == Arena::psum)
    unwrite(a);
  if (a.input.empty())
    return;
  if (a.arena != Arena::hbm)
    fail("input " + a.input + " is not in hbm");
  std::string path = inputs + "/" + a.input + ".bin";
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    fail("cannot read " + path);
  if (static_cast<int64_t>(in.tellg()) != a.rowBytes())
    fail(path + " holds " + std::to_string(in.tellg()) + " bytes, input " + a.input + " " +
         std::to_string(a.rowBytes()));
  in.seekg(0);
  in.read(reinterpret_cast<char *>(at(Arena::hbm, 0, a.byte)), a.rowBytes());
}

void Accelerator::check(const View &v) const {
  std::string where = v.address();
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
}

View Accelerator::resolve(const View &view) {
  View v = view;
  if (v.dyn) {
    View inner = resolve(*v.dyn->view);
    if (inner.dtype != Dtype::i32 || inner.count() != 1)
      fail("a +s() offset reads one i32; a vector offset is milestone M3");
    int32_t offset;
    std::memcpy(&offset, at(inner.arena, inner.partition, inner.byte), sizeof offset);
    v.byte += offset * v.dyn->scale * width(v.dtype);
    v.dyn.reset();
  }
  check(v);
  return v;
}

void Accelerator::wrote(const View &v) {
  if (v.arena == Arena::psum)
    mark(v, 1);
}

std::vector<bool> Accelerator::started(const View &v) const {
  std::vector<bool> out;
  int64_t k = width(v.dtype);
  v.each([&](int64_t p, int64_t e) {
    auto first = written.begin() + p * kPsumRow + e * k;
    out.push_back(v.arena == Arena::psum &&
                  std::all_of(first, first + k, [](uint8_t b) { return b != 0; }));
  });
  return out;
}

void Accelerator::restart(const View &v) {
  if (v.arena != Arena::psum)
    return;
  // The newest allocation holding it, past any dead ones.
  if (const Alloc *a = holder(allocs, v)) {
    unwrite(*a);
    return;
  }
  mark(v, 0);
}

void Accelerator::mark(const View &v, uint8_t value) {
  int64_t k = width(v.dtype);
  v.each(
      [&](int64_t p, int64_t e) { std::fill_n(written.begin() + p * kPsumRow + e * k, k, value); });
}

void Accelerator::unwrite(const Alloc &a) {
  for (int64_t p = a.partition; p < a.partition + a.partitions(); ++p)
    std::fill_n(written.begin() + p * kPsumRow + a.byte, a.rowBytes(), 0);
}

void Accelerator::store(const Output &o, const std::string &dir) const {
  auto it = allocs.find(o.id);
  if (it == allocs.end())
    fail("output " + o.name + " names no allocation %" + std::to_string(o.id));
  const Alloc &a = it->second;
  std::string path = dir + "/" + o.name + ".bin";
  std::ofstream out(path, std::ios::binary);
  const std::vector<uint8_t> &bytes = a.arena == Arena::hbm    ? hbm
                                      : a.arena == Arena::sbuf ? sbuf
                                                               : psum;
  for (int64_t p = a.partition; p < a.partition + a.partitions(); ++p)
    out.write(reinterpret_cast<const char *>(bytes.data() + p * rowBytes(a.arena) + a.byte),
              a.rowBytes());
  if (!out)
    fail("cannot write " + path);
}

} // namespace nki
