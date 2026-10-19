#include "window.h"

#include <algorithm>
#include <numeric>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

namespace fsim {
namespace {

// Past this many intervals a window has no bytes.
constexpr int64_t kCap = 4096;

// An element's bits, or 0 unless an int or float.
int64_t bits(mlir::Type type) {
  return type.isIntOrFloat() ? type.getIntOrFloatBitWidth() : 0;
}

// The box over a static tensor type, width set; none otherwise.
std::optional<Window> over(mlir::RankedTensorType type) {
  int64_t width = bits(type.getElementType());
  if (!type.hasStaticShape() || width == 0 || width % 8 != 0)
    return std::nullopt;
  Window window;
  window.dims.assign(type.getShape().begin(), type.getShape().end());
  window.width = width / 8;
  return window;
}

// A single element's stride is 1, so equal sets compare equal.
void normalize(Window &window) {
  for (size_t d = 0; d < window.dims.size(); ++d)
    if (window.count[d] == 1)
      window.stride[d] = 1;
}

// Whether the box selects no element.
bool empty(const Window &window) {
  return llvm::is_contained(window.count, 0);
}

// The bytes in intervals.
int64_t total(const Intervals &x) {
  int64_t size = 0;
  for (auto [begin, end] : x)
    size += end - begin;
  return size;
}

// The bytes x and y share.
int64_t overlap(const Intervals &x, const Intervals &y) {
  int64_t size = 0;
  for (size_t i = 0, j = 0; i < x.size() && j < y.size();) {
    size +=
        std::max<int64_t>(0, std::min(x[i].second, y[j].second) - std::max(x[i].first, y[j].first));
    if (x[i].second < y[j].second)
      ++i;
    else
      ++j;
  }
  return size;
}

// The same bytes over the fewest dims: bytes as a trailing dim, unit
// dims dropped, each dim merged into the whole dim inside it.
Window merged(const Window &window) {
  Window out;
  out.width = 1;
  out.dims = {window.width};
  out.start = {0};
  out.count = {window.width};
  out.stride = {1};
  for (size_t d = window.dims.size(); d-- > 0;) {
    int64_t size = window.dims[d];
    if (size == 1)
      continue;
    int64_t inner = out.dims.back();
    bool whole = out.start.back() == 0 && out.count.back() == inner && out.stride.back() == 1;
    if (whole && window.stride[d] == 1) {
      out.dims.back() = size * inner;
      out.start.back() = window.start[d] * inner;
      out.count.back() = window.count[d] * inner;
      continue;
    }
    out.dims.push_back(size);
    out.start.push_back(window.start[d]);
    out.count.push_back(window.count[d]);
    out.stride.push_back(window.stride[d]);
  }
  // Outer dims first.
  std::reverse(out.dims.begin(), out.dims.end());
  std::reverse(out.start.begin(), out.start.end());
  std::reverse(out.count.begin(), out.count.end());
  std::reverse(out.stride.begin(), out.stride.end());
  return out;
}

// Boxes over one shape, dim by dim; Other when undecided.
std::optional<Relation> relateBoxes(const Window &a, const Window &b) {
  if (a.dims != b.dims || a.width != b.width)
    return std::nullopt;
  bool unit = true, equal = true, inside = true;
  for (size_t d = 0; d < a.dims.size(); ++d) {
    int64_t s1 = a.start[d], n1 = a.count[d], t1 = a.stride[d];
    int64_t s2 = b.start[d], n2 = b.count[d], t2 = b.stride[d];
    int64_t l1 = s1 + (n1 - 1) * t1, l2 = s2 + (n2 - 1) * t2;
    // One disjoint dim makes the boxes disjoint.
    if (l1 < s2 || l2 < s1 || (s1 - s2) % std::gcd(t1, t2) != 0)
      return Relation::Disjoint;
    unit &= t1 == 1 && t2 == 1;
    equal &= s1 == s2 && n1 == n2 && t1 == t2;
    inside &= s1 >= s2 && l1 <= l2 && (s1 - s2) % t2 == 0 && (n1 == 1 || t1 % t2 == 0);
  }
  if (equal)
    return Relation::Equal;
  if (inside)
    return Relation::Inside;
  // Unit strides overlapping in every dim overlap.
  if (unit)
    return Relation::Other;
  return std::nullopt;
}

} // namespace

std::optional<Window> Window::of(mlir::stablehlo::SliceOp slice) {
  std::optional<Window> window = over(slice.getOperand().getType());
  if (!window)
    return std::nullopt;
  window->start.assign(slice.getStartIndices().begin(), slice.getStartIndices().end());
  window->stride.assign(slice.getStrides().begin(), slice.getStrides().end());
  for (size_t d = 0; d < window->dims.size(); ++d) {
    int64_t span = slice.getLimitIndices()[d] - window->start[d];
    window->count.push_back(
        std::max<int64_t>(0, (span + window->stride[d] - 1) / window->stride[d]));
  }
  normalize(*window);
  return window;
}

std::optional<Window> Window::of(mlir::stablehlo::DynamicUpdateSliceOp update) {
  std::optional<Window> window = over(update.getType());
  if (!window)
    return std::nullopt;
  llvm::ArrayRef<int64_t> sizes = update.getUpdate().getType().getShape();
  for (auto [d, index] : llvm::enumerate(update.getStartIndices())) {
    mlir::DenseIntElementsAttr value;
    if (!mlir::matchPattern(index, mlir::m_Constant(&value)))
      return std::nullopt;
    llvm::APInt raw = *value.begin();
    int64_t start =
        value.getElementType().isUnsignedInteger() ? raw.getZExtValue() : raw.getSExtValue();
    // The op clamps the start so the update fits.
    window->start.push_back(std::clamp<int64_t>(start, 0, window->dims[d] - sizes[d]));
    window->count.push_back(sizes[d]);
    window->stride.push_back(1);
  }
  normalize(*window);
  return window;
}

const Intervals *Window::bytes() const {
  if (computed_)
    return bytes_ ? &*bytes_ : nullptr;
  computed_ = true;
  Intervals out;
  if (empty(*this)) {
    bytes_ = std::move(out);
    return &*bytes_;
  }

  // Bytes per step of each dim, row-major.
  size_t rank = dims.size();
  llvm::SmallVector<int64_t> pitch(rank);
  int64_t step = width;
  for (size_t d = rank; d-- > 0;) {
    pitch[d] = step;
    step *= dims[d];
  }

  // Inner dims [outer, rank) make one contiguous run.
  size_t outer = rank;
  int64_t run = width;
  while (outer > 0 && stride[outer - 1] == 1) {
    --outer;
    run *= count[outer];
    if (count[outer] != dims[outer])
      break;
  }
  int64_t runs = 1;
  for (size_t d = 0; d < outer; ++d)
    runs *= count[d];
  if (runs > kCap)
    return nullptr;

  // Walks the outer dims as an odometer, ascending.
  int64_t base = 0;
  for (size_t d = 0; d < rank; ++d)
    base += start[d] * pitch[d];
  llvm::SmallVector<int64_t> at(outer, 0);
  out.reserve(runs);
  for (int64_t r = 0; r < runs; ++r) {
    int64_t begin = base;
    for (size_t d = 0; d < outer; ++d)
      begin += at[d] * stride[d] * pitch[d];
    if (!out.empty() && out.back().second == begin)
      out.back().second += run;
    else
      out.emplace_back(begin, begin + run);
    for (size_t d = outer; d-- > 0;) {
      if (++at[d] < count[d])
        break;
      at[d] = 0;
    }
  }
  bytes_ = std::move(out);
  return &*bytes_;
}

Relation relate(const Window &a, const Window &b) {
  if (empty(a) || empty(b))
    return Relation::Disjoint;
  if (std::optional<Relation> boxes = relateBoxes(a, b))
    return *boxes;
  if (std::optional<Relation> boxes = relateBoxes(merged(a), merged(b)))
    return *boxes;
  const Intervals *x = a.bytes();
  const Intervals *y = b.bytes();
  if (!x || !y)
    return Relation::Other;
  int64_t both = overlap(*x, *y), size = total(*x);
  if (both == 0)
    return Relation::Disjoint;
  if (both == size)
    return size == total(*y) ? Relation::Equal : Relation::Inside;
  return Relation::Other;
}

bool covered(const Intervals &a, const Intervals &u) {
  for (auto [begin, end] : a) {
    // The last interval of u starting at or before begin.
    auto it = std::upper_bound(u.begin(), u.end(), begin,
                               [](int64_t value, const auto &x) { return value < x.first; });
    if (it == u.begin() || std::prev(it)->second < end)
      return false;
  }
  return true;
}

void unite(Intervals &u, const Intervals &a) {
  Intervals merged;
  merged.reserve(u.size() + a.size());
  std::merge(u.begin(), u.end(), a.begin(), a.end(), std::back_inserter(merged));
  u.clear();
  for (auto [begin, end] : merged) {
    if (!u.empty() && begin <= u.back().second)
      u.back().second = std::max(u.back().second, end);
    else
      u.emplace_back(begin, end);
  }
}

bool isView(mlir::Operation *op) {
  if (mlir::isa<mlir::stablehlo::ReshapeOp>(op))
    return true;
  auto bitcast = mlir::dyn_cast<mlir::stablehlo::BitcastConvertOp>(op);
  if (!bitcast)
    return false;
  int64_t from = bits(bitcast.getOperand().getType().getElementType());
  return from != 0 && from == bits(bitcast.getType().getElementType());
}

} // namespace fsim
