#include "stablehlo/Module.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace nki {

namespace {

[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error(what);
}

std::string join(const std::vector<int64_t> &v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i)
    s += (i ? ", " : "") + std::to_string(v[i]);
  return s;
}

std::string i64s(const std::vector<int64_t> &v) {
  return v.empty() ? "array<i64>" : "array<i64: " + join(v) + ">";
}

std::string hex(uint32_t bits, int digits) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "0x%0*X", digits, bits);
  return buf;
}

// f32 to bf16 bits, round to nearest even.
uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if (std::isnan(value))
    return 0x7FC0;
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// An integer wrapped to the dtype's width, as a literal.
std::string wrapped(int64_t value, Dtype dtype) {
  switch (dtype) {
  case Dtype::u8:
    return std::to_string(static_cast<uint8_t>(value));
  case Dtype::u16:
    return std::to_string(static_cast<uint16_t>(value));
  case Dtype::u32:
    return std::to_string(static_cast<uint32_t>(value));
  default:
    return std::to_string(static_cast<int32_t>(value));
  }
}

// The dtype's integer range.
std::pair<double, double> range(Dtype dtype) {
  switch (dtype) {
  case Dtype::u8:
    return {0, 255};
  case Dtype::u16:
    return {0, 65535};
  case Dtype::u32:
    return {0, 4294967295.0};
  default:
    return {-2147483648.0, 2147483647.0};
  }
}

bool isFloat(const std::string &elem) {
  return elem == "f32" || elem == "bf16";
}

// A view's geometry over its arena's [P, E] element grid.
// An .ap() element: offset + sum of index_k * stride_k.
struct Geometry {
  int64_t p0 = 0, ps = 1, pc = 1; // the partitions read, broadcast folded to one
  bool spread = false;            // the partitions broadcast
  int64_t e0 = 0;                 // the first element
  std::vector<Entry> kept;        // free dims past 1 that advance
  std::vector<int64_t> at;        // each kept dim's position in the shape
  bool broadcast = false;         // a free dim of stride 0
  bool nested = false;            // the kept dims tile one window
  int64_t window = 1;             // the window's elements
};

Geometry geometry(const View &v, int64_t elements) {
  Geometry g;
  Entry p = v.partitions();
  g.p0 = v.partition;
  g.spread = p.stride == 0 && p.count > 1;
  g.pc = g.spread ? 1 : p.count;
  g.ps = g.pc == 1 ? 1 : p.stride;
  g.e0 = v.byte / width(v.dtype);
  std::vector<Entry> free = v.free();
  for (size_t d = 0; d < free.size(); ++d) {
    if (free[d].count == 1)
      continue;
    if (free[d].stride == 0) {
      g.broadcast = true;
      continue;
    }
    g.kept.push_back(free[d]);
    g.at.push_back(static_cast<int64_t>(d) + 1);
  }
  // Nested: positive strides, each dividing the outer, disjoint.
  g.nested = g.ps > 0;
  for (size_t i = 0; i < g.kept.size(); ++i) {
    const Entry &e = g.kept[i];
    if (e.stride <= 0)
      g.nested = false;
    else if (i + 1 < g.kept.size()) {
      const Entry &in = g.kept[i + 1];
      if (in.stride <= 0 || e.stride % in.stride != 0 || in.count * in.stride > e.stride)
        g.nested = false;
    }
  }
  g.window = g.kept.empty() ? 1 : g.kept[0].count * g.kept[0].stride;
  if (g.e0 + g.window > elements)
    g.nested = false;
  return g;
}

// Nested strides delinearize the window: [count_0, s_0/s_1, ..., s_n].
// The limits keep each count; the innermost index is 0.
void delinearize(const Geometry &g, std::vector<int64_t> &dims, std::vector<int64_t> &limits) {
  dims = {g.pc, g.kept[0].count};
  limits = {g.pc, g.kept[0].count};
  for (size_t i = 1; i < g.kept.size(); ++i) {
    dims.push_back(g.kept[i - 1].stride / g.kept[i].stride);
    limits.push_back(g.kept[i].count);
  }
  dims.push_back(g.kept.back().stride);
  limits.push_back(1);
}

// Flat indices of the elements the kept dims reach.
std::vector<int64_t> indices(const Geometry &g, int64_t elements) {
  std::vector<int64_t> out;
  View walk;
  walk.arena = Arena::sbuf;
  walk.dtype = Dtype::u8;
  walk.partition = g.p0;
  walk.byte = g.e0;
  walk.pattern.push_back({g.ps, g.pc});
  walk.pattern.insert(walk.pattern.end(), g.kept.begin(), g.kept.end());
  walk.each([&](int64_t p, int64_t e) { out.push_back(p * elements + e); });
  return out;
}

} // namespace

std::string Type::str() const {
  std::string s = "tensor<";
  for (int64_t d : dims)
    s += std::to_string(d) + "x";
  return s + elem + ">";
}

int64_t Type::count() const {
  int64_t n = 1;
  for (int64_t d : dims)
    n *= d;
  return n;
}

Module::Module(std::vector<Type> state) : state_(std::move(state)) {
  reset();
}

void Module::reset() {
  types_ = state_;
  slots_.clear();
  for (size_t i = 0; i < state_.size(); ++i)
    slots_.push_back(static_cast<int>(i));
  body_.str("");
  bitcasts_.clear();
}

std::string Module::text() {
  if (types_.size() == slots_.size())
    return "";
  std::ostringstream m;
  m << "module {\n  func.func @main(";
  for (size_t i = 0; i < slots_.size(); ++i)
    m << (i ? ", " : "") << "%v" << i << ": " << types_[i].str();
  m << ") -> (";
  for (size_t i = 0; i < slots_.size(); ++i)
    m << (i ? ", " : "") << types_[slots_[i]].str();
  m << ") {\n" << body_.str() << "    \"func.return\"(";
  for (size_t i = 0; i < slots_.size(); ++i)
    m << (i ? ", " : "") << "%v" << slots_[i];
  m << ") : (";
  for (size_t i = 0; i < slots_.size(); ++i)
    m << (i ? ", " : "") << types_[slots_[i]].str();
  m << ") -> ()\n  }\n}\n";
  reset();
  return m.str();
}

int Module::emit(const std::string &op, const std::vector<int> &operands, const std::string &attrs,
                 Type type, const std::string &region) {
  int id = static_cast<int>(types_.size());
  body_ << "    %v" << id << " = \"stablehlo." << op << "\"(";
  for (size_t i = 0; i < operands.size(); ++i)
    body_ << (i ? ", " : "") << "%v" << operands[i];
  body_ << ")" << region;
  if (!attrs.empty())
    body_ << " {" << attrs << "}";
  body_ << " : (";
  for (size_t i = 0; i < operands.size(); ++i)
    body_ << (i ? ", " : "") << types_[operands[i]].str();
  body_ << ") -> " << type.str() << "\n";
  types_.push_back(std::move(type));
  return id;
}

// ---- Memory ----

int Module::bitcastTo(Arena arena, Dtype dtype) {
  int slot = static_cast<int>(arena);
  auto it = bitcasts_.find({slot, dtype});
  if (it != bitcasts_.end())
    return it->second;
  int bytes = slots_[slot];
  std::vector<int64_t> d = types_[bytes].dims;
  int64_t k = width(dtype);
  int value = dtype == Dtype::u8
                  ? bytes
                  : bitcast(reshape(bytes, {d[0], d[1] / k, k}), {mlir(dtype), {d[0], d[1] / k}});
  bitcasts_[{slot, dtype}] = value;
  return value;
}

void Module::retype(Arena arena, Dtype dtype, int value) {
  int slot = static_cast<int>(arena);
  std::vector<int64_t> d = types_[value].dims;
  int64_t k = width(dtype);
  slots_[slot] = dtype == Dtype::u8
                     ? value
                     : reshape(bitcast(value, {"ui8", {d[0], d[1], k}}), {d[0], d[1] * k});
  std::erase_if(bitcasts_, [&](const auto &entry) { return entry.first.first == slot; });
  bitcasts_[{slot, dtype}] = value;
}

int Module::read(const View &v) {
  int t = bitcastTo(v.arena, v.dtype);
  std::vector<int64_t> d = types_[t].dims;
  Geometry g = geometry(v, d[1]);
  // The kept elements, [pc, kept counts...].
  std::vector<int64_t> tile = {g.pc};
  for (const Entry &e : g.kept)
    tile.push_back(e.count);
  int out;
  if (g.nested && g.kept.size() == 1) {
    // One level: a strided slice takes index * stride.
    const Entry &e = g.kept[0];
    out =
        slice(t, {g.p0, g.e0}, {g.p0 + (g.pc - 1) * g.ps + 1, g.e0 + (e.count - 1) * e.stride + 1},
              {g.ps, e.stride});
  } else if (g.nested) {
    int w = slice(t, {g.p0, g.e0}, {g.p0 + (g.pc - 1) * g.ps + 1, g.e0 + g.window}, {g.ps, 1});
    if (g.kept.empty()) {
      out = reshape(w, tile);
    } else {
      std::vector<int64_t> dims, limits;
      delinearize(g, dims, limits);
      int s = slice(reshape(w, dims), std::vector<int64_t>(dims.size(), 0), limits,
                    std::vector<int64_t>(dims.size(), 1));
      out = reshape(s, tile);
    }
  } else {
    out = reshape(gather(reshape(t, {d[0] * d[1]}), indices(g, d[1])), tile);
  }
  // To the full shape: broadcasts expand, unit dims return.
  std::vector<int64_t> full = {v.partitions().count};
  for (const Entry &e : v.free())
    full.push_back(e.count);
  if (g.spread || g.broadcast) {
    std::vector<int64_t> map = {0};
    map.insert(map.end(), g.at.begin(), g.at.end());
    out = broadcast(out, full, map);
  }
  return reshape(out, v.shape());
}

void Module::write(const View &v, int value) {
  if (types_[value].count() != v.count() || types_[value].elem != mlir(v.dtype))
    fail("a write of " + types_[value].str() + " through a view of " + std::to_string(v.count()) +
         " " + mlir(v.dtype));
  int t = bitcastTo(v.arena, v.dtype);
  std::vector<int64_t> d = types_[t].dims;
  Geometry g = geometry(v, d[1]);
  if (g.spread || g.broadcast)
    fail("a write through a broadcast view");
  if (g.nested && g.ps == 1) {
    int win;
    std::vector<int64_t> dims, limits;
    if (!g.kept.empty())
      delinearize(g, dims, limits);
    // A dense window takes the value whole, else a read-update.
    bool dense = g.kept.empty() ||
                 (dims.back() == 1 && std::equal(dims.begin(), dims.end() - 1, limits.begin()));
    if (dense) {
      win = reshape(value, {g.pc, g.window});
    } else {
      int w = slice(t, {g.p0, g.e0}, {g.p0 + g.pc, g.e0 + g.window}, {1, 1});
      int u =
          update(reshape(w, dims), reshape(value, limits), std::vector<int64_t>(dims.size(), 0));
      win = reshape(u, {g.pc, g.window});
    }
    retype(v.arena, v.dtype, update(t, win, {g.p0, g.e0}));
    return;
  }
  int flat = scatter(reshape(t, {d[0] * d[1]}), indices(g, d[1]), reshape(value, {v.count()}));
  retype(v.arena, v.dtype, reshape(flat, d));
}

void Module::lanes(Slot slot, int64_t first, int64_t count) const {
  int64_t partitions = state_[static_cast<int>(slot)].dims[0];
  if (first < 0 || first + count > partitions)
    fail("reduce register lanes " + std::to_string(first) + "+" + std::to_string(count) + " past " +
         std::to_string(partitions));
}

int Module::readLanes(Slot slot, int64_t first, int64_t count) {
  lanes(slot, first, count);
  return slice(slots_[static_cast<int>(slot)], {first, 0}, {first + count, 1}, {1, 1});
}

void Module::writeLanes(Slot slot, int64_t first, int value) {
  int64_t count = types_[value].count();
  lanes(slot, first, count);
  int s = static_cast<int>(slot);
  slots_[s] = update(slots_[s], reshape(value, {count, 1}), {first, 0});
}

// ---- Constants ----

int Module::constant(const Type &type, const std::string &literal) {
  int id = static_cast<int>(types_.size());
  body_ << "    %v" << id << " = \"stablehlo.constant\"() {value = dense<" << literal
        << "> : " << type.str() << "} : () -> " << type.str() << "\n";
  types_.push_back(type);
  return id;
}

int Module::f32(double value, std::vector<int64_t> dims) {
  return constant({"f32", std::move(dims)},
                  hex(std::bit_cast<uint32_t>(static_cast<float>(value)), 8));
}

int Module::fill(Dtype dtype, const Number &value, std::vector<int64_t> dims) {
  std::string literal;
  if (dtype == Dtype::f32)
    literal = hex(std::bit_cast<uint32_t>(static_cast<float>(value.value)), 8);
  else if (dtype == Dtype::bf16)
    literal = hex(bf16(static_cast<float>(value.value)), 4);
  else
    literal = wrapped(value.integer ? *value.integer : static_cast<int64_t>(value.value), dtype);
  return constant({mlir(dtype), std::move(dims)}, literal);
}

int Module::mask(const std::vector<bool> &bits, std::vector<int64_t> dims) {
  std::string literal = "[";
  for (size_t i = 0; i < bits.size(); ++i)
    literal += std::string(i ? ", " : "") + (bits[i] ? "true" : "false");
  return reshape(constant({"i1", {static_cast<int64_t>(bits.size())}}, literal + "]"),
                 std::move(dims));
}

int Module::index(int64_t value) {
  return constant({"i32", {}}, std::to_string(value));
}

// ---- Shape ops ----

int Module::reshape(int value, std::vector<int64_t> dims) {
  if (types_[value].dims == dims)
    return value;
  Type to{types_[value].elem, dims};
  if (to.count() != types_[value].count())
    fail("a reshape of " + types_[value].str() + " to " + to.str());
  return emit("reshape", {value}, "", {types_[value].elem, std::move(dims)});
}

int Module::broadcast(int value, std::vector<int64_t> dims, std::vector<int64_t> map) {
  if (types_[value].dims == dims)
    return value;
  return emit("broadcast_in_dim", {value}, "broadcast_dimensions = " + i64s(map),
              {types_[value].elem, std::move(dims)});
}

int Module::slice(int value, std::vector<int64_t> start, std::vector<int64_t> limit,
                  std::vector<int64_t> strides) {
  std::vector<int64_t> dims;
  for (size_t i = 0; i < start.size(); ++i)
    dims.push_back((limit[i] - start[i] + strides[i] - 1) / strides[i]);
  if (dims == types_[value].dims)
    return value;
  return emit("slice", {value},
              "start_indices = " + i64s(start) + ", limit_indices = " + i64s(limit) +
                  ", strides = " + i64s(strides),
              {types_[value].elem, std::move(dims)});
}

int Module::update(int operand, int tile, const std::vector<int64_t> &start) {
  // A full tile at the origin replaces the operand.
  bool origin = std::all_of(start.begin(), start.end(), [](int64_t s) { return s == 0; });
  if (origin && types_[tile].dims == types_[operand].dims)
    return tile;
  std::vector<int> operands = {operand, tile};
  for (int64_t s : start)
    operands.push_back(index(s));
  return emit("dynamic_update_slice", operands, "", types_[operand]);
}

int Module::transpose(int value, std::vector<int64_t> permutation) {
  std::vector<int64_t> dims;
  for (int64_t p : permutation)
    dims.push_back(types_[value].dims[p]);
  return emit("transpose", {value}, "permutation = " + i64s(permutation),
              {types_[value].elem, std::move(dims)});
}

int Module::gather(int flat, const std::vector<int64_t> &indices) {
  std::string literal = "[";
  for (size_t i = 0; i < indices.size(); ++i)
    literal += (i ? ", [" : "[") + std::to_string(indices[i]) + "]";
  int64_t n = static_cast<int64_t>(indices.size());
  int idx = constant({"i32", {n, 1}}, literal + "]");
  return emit("gather", {flat, idx},
              "dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], "
              "start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, "
              "slice_sizes = array<i64: 1>",
              {types_[flat].elem, {n}});
}

int Module::scatter(int flat, const std::vector<int64_t> &indices, int updates) {
  std::string literal = "[";
  for (size_t i = 0; i < indices.size(); ++i)
    literal += (i ? ", [" : "[") + std::to_string(indices[i]) + "]";
  int64_t n = static_cast<int64_t>(indices.size());
  int idx = constant({"i32", {n, 1}}, literal + "]");
  // The update keeps the written element.
  std::string id = std::to_string(types_.size());
  std::string scalar = Type{types_[flat].elem, {}}.str();
  std::string region = " ({\n    ^bb0(%s" + id + "a: " + scalar + ", %s" + id + "b: " + scalar +
                       "):\n      \"stablehlo.return\"(%s" + id + "b) : (" + scalar +
                       ") -> ()\n    })";
  return emit("scatter", {flat, idx, updates},
              "indices_are_sorted = false, scatter_dimension_numbers = "
              "#stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = "
              "[0], index_vector_dim = 1>, unique_indices = false",
              types_[flat], region);
}

// ---- Element ops ----

int Module::bitcast(int value, const Type &type) {
  return emit("bitcast_convert", {value}, "", type);
}

int Module::convert(int value, const std::string &elem) {
  if (types_[value].elem == elem)
    return value;
  return emit("convert", {value}, "", {elem, types_[value].dims});
}

int Module::unary(const std::string &op, int value) {
  return emit(op, {value}, "", types_[value]);
}

int Module::binary(const std::string &op, int lhs, int rhs) {
  return emit(op, {lhs, rhs}, "", types_[lhs]);
}

int Module::compare(const std::string &direction, int lhs, int rhs) {
  return emit("compare", {lhs, rhs},
              "comparison_direction = #stablehlo<comparison_direction " + direction + ">",
              {"i1", types_[lhs].dims});
}

int Module::select(int pred, int onTrue, int onFalse) {
  return emit("select", {pred, onTrue, onFalse}, "", types_[onTrue]);
}

int Module::reduce(int value, const std::string &op, std::vector<int64_t> axes) {
  static const std::map<std::string, std::string> inits = {{"add", "0x00000000"},
                                                           {"multiply", "0x3F800000"},
                                                           {"maximum", "0xFF800000"},
                                                           {"minimum", "0x7F800000"}};
  int init = constant({"f32", {}}, inits.at(op));
  std::vector<int64_t> kept;
  const std::vector<int64_t> &from = types_[value].dims;
  for (size_t i = 0; i < from.size(); ++i)
    if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) == axes.end())
      kept.push_back(from[i]);
  std::string id = std::to_string(types_.size());
  std::string r = "%r" + id;
  std::string region = " ({\n    ^bb0(" + r + "a: tensor<f32>, " + r + "b: tensor<f32>):\n      " +
                       r + "c = \"stablehlo." + op + "\"(" + r + "a, " + r +
                       "b) : (tensor<f32>, tensor<f32>) -> tensor<f32>\n" +
                       "      \"stablehlo.return\"(" + r + "c) : (tensor<f32>) -> ()\n    })";
  return emit("reduce", {value, init}, "dimensions = " + i64s(axes), {"f32", kept}, region);
}

int Module::dot(int lhs, int rhs) {
  return emit("dot_general", {lhs, rhs},
              "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], "
              "rhs_contracting_dimensions = [0]>, precision_config = "
              "[#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]",
              {"f32", {types_[lhs].dims[1], types_[rhs].dims[1]}});
}

int Module::toF32(int value) {
  return convert(value, "f32");
}

int Module::toDtype(int value, Dtype dtype) {
  const std::string &elem = types_[value].elem;
  if (elem == mlir(dtype))
    return value;
  if (!integral(dtype) || !isFloat(elem))
    return convert(value, mlir(dtype));
  // Round to nearest even, saturate, NaN to the maximum.
  std::vector<int64_t> d = types_[value].dims;
  int x = toF32(value);
  int r = unary("round_nearest_even", x);
  auto [lo, hi] = range(dtype);
  float top = static_cast<float>(hi);
  if (top > hi)
    top = std::nextafter(top, 0.0f);
  int clamped = binary("minimum", binary("maximum", r, f32(lo, d)), f32(top, d));
  int over = binary("or", compare("GT", r, f32(top, d)), compare("NE", x, x));
  Number max{hi, static_cast<int64_t>(hi)};
  return select(over, fill(dtype, max, d), convert(clamped, mlir(dtype)));
}

std::pair<int, int> Module::broadcastPair(int lhs, int rhs) {
  std::vector<int64_t> a = types_[lhs].dims, b = types_[rhs].dims;
  size_t rank = std::max(a.size(), b.size());
  a.insert(a.begin(), rank - a.size(), 1);
  b.insert(b.begin(), rank - b.size(), 1);
  std::vector<int64_t> dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
      fail("operands " + types_[lhs].str() + " and " + types_[rhs].str() + " do not broadcast");
    dims[i] = std::max(a[i], b[i]);
  }
  // Pad to the rank, then expand the unit dims.
  std::vector<int64_t> all;
  for (size_t i = 0; i < rank; ++i)
    all.push_back(static_cast<int64_t>(i));
  auto to = [&](int value, const std::vector<int64_t> &padded) {
    return broadcast(reshape(value, padded), dims, all);
  };
  return {to(lhs, a), to(rhs, b)};
}

} // namespace nki
