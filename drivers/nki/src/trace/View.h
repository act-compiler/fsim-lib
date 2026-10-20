// A View is one operand; a Dyn, its run-time offset.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nki {

enum class Arena { hbm, sbuf, psum };
enum class Dtype { bf16, f32, i32, u8, u16, u32 };

// Bytes per element.
inline int64_t width(Dtype dtype) {
  switch (dtype) {
  case Dtype::u8:
    return 1;
  case Dtype::bf16:
  case Dtype::u16:
    return 2;
  default:
    return 4;
  }
}

// Whether the dtype is an integer.
inline bool integral(Dtype dtype) {
  return dtype != Dtype::bf16 && dtype != Dtype::f32;
}

// The arena as a record spells it.
inline const char *name(Arena arena) {
  static const char *const names[] = {"hbm", "sbuf", "psum"};
  return names[static_cast<int>(arena)];
}

// One pattern entry: a stride and a count.
struct Entry {
  int64_t stride = 0;
  int64_t count = 0;
};

// The lowest and highest index the entries reach.
inline std::pair<int64_t, int64_t> span(int64_t first, const std::vector<Entry> &entries) {
  int64_t lo = first, hi = first;
  for (const Entry &e : entries) {
    int64_t reach = (e.count - 1) * e.stride;
    (reach < 0 ? lo : hi) += reach;
  }
  return {lo, hi};
}

struct View;

// Run-time offset: an i32 view's value, times scale.
struct Dyn {
  std::shared_ptr<const View> view;
  int64_t scale = 0;
};

struct View {
  Arena arena = Arena::hbm;
  int64_t partition = 0;
  int64_t byte = 0;
  Dtype dtype = Dtype::u8;
  std::vector<Entry> pattern;
  std::optional<Dyn> dyn;

  // Whether the first entry is the partition dimension.
  bool partitioned() const {
    return arena != Arena::hbm;
  }
  // The partition entry; hbm has one partition.
  Entry partitions() const {
    return partitioned() ? pattern.front() : Entry{0, 1};
  }
  // The entries within a partition, outer first.
  std::vector<Entry> free() const {
    return {pattern.begin() + (partitioned() ? 1 : 0), pattern.end()};
  }
  // The first byte as arena@partition,byte.
  std::string address() const {
    return std::string(name(arena)) + "@" + std::to_string(partition) + "," + std::to_string(byte);
  }
  // The operand's shape: the counts.
  std::vector<int64_t> shape() const;
  // Elements in the operand.
  int64_t count() const;
  // Calls f(partition, element) per element, row-major order.
  template <class F> void each(F f) const;
};

inline std::vector<int64_t> View::shape() const {
  std::vector<int64_t> dims;
  for (const Entry &e : pattern)
    dims.push_back(e.count);
  return dims;
}

inline int64_t View::count() const {
  int64_t n = 1;
  for (const Entry &e : pattern)
    n *= e.count;
  return n;
}

// The element is in dtype units from its row's start.
template <class F> void View::each(F f) const {
  Entry p = partitions();
  std::vector<Entry> dims = free();
  int64_t first = byte / width(dtype);
  for (int64_t i = 0; i < p.count; ++i) {
    // An odometer over the free dims, innermost fastest.
    std::vector<int64_t> at(dims.size(), 0);
    int64_t total = 1;
    for (const Entry &e : dims)
      total *= e.count;
    for (int64_t n = 0; n < total; ++n) {
      int64_t element = first;
      for (size_t d = 0; d < dims.size(); ++d)
        element += at[d] * dims[d].stride;
      f(partition + i * p.stride, element);
      for (size_t d = dims.size(); d-- > 0;) {
        if (++at[d] < dims[d].count)
          break;
        at[d] = 0;
      }
    }
  }
}

} // namespace nki
