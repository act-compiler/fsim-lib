// The StableHLO module: state parameters, view reads and writes as
// ops, and the text that text() returns.
// A value is an index; ops() counts the parameters too.
#pragma once

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "trace/Fields.h"

namespace nki {

// One value's element type and dims.
struct Type {
  std::string elem;
  std::vector<int64_t> dims;

  std::string str() const;
  int64_t count() const;
};

// The StableHLO element type.
inline const char *mlir(Dtype dtype) {
  static const char *const names[] = {"bf16", "f32", "i32", "ui8", "ui16", "ui32"};
  return names[static_cast<int>(dtype)];
}

// The state parameters, in @main's order; arenas first.
enum class Slot { hbm, sbuf, psum, reduceRegs };

class Module {
public:
  // The arenas are ui8[P, R]; reduce registers f32[P, 1].
  explicit Module(std::vector<Type> state);

  // Values in the module, the parameters counted.
  size_t ops() const {
    return types_.size();
  }
  // Ends the module, returns its text, starts afresh.
  // The text is empty when the module holds no ops.
  std::string text();

  // The view's elements, [counts...] in its dtype.
  int read(const View &view);
  // Writes the view's dtype and count, in flat order.
  void write(const View &view, int value);
  // Reduce register lanes [first, first + count), f32[count, 1].
  int readLanes(Slot slot, int64_t first, int64_t count);
  void writeLanes(Slot slot, int64_t first, int value);

  const Type &type(int value) const {
    return types_[value];
  }
  std::vector<int64_t> shape(int value) const {
    return types_[value].dims;
  }

  // Constants.
  int constant(const Type &type, const std::string &literal);
  int f32(double value, std::vector<int64_t> dims);
  int fill(Dtype dtype, const Number &value, std::vector<int64_t> dims);
  int mask(const std::vector<bool> &bits, std::vector<int64_t> dims);

  // Shape ops.
  int reshape(int value, std::vector<int64_t> dims);
  int broadcast(int value, std::vector<int64_t> dims, std::vector<int64_t> map);
  int slice(int value, std::vector<int64_t> start, std::vector<int64_t> limit,
            std::vector<int64_t> strides);
  int update(int operand, int tile, const std::vector<int64_t> &start);
  int transpose(int value, std::vector<int64_t> permutation);
  int gather(int flat, const std::vector<int64_t> &indices);
  int scatter(int flat, const std::vector<int64_t> &indices, int updates);

  // Element ops.
  int bitcast(int value, const Type &type);
  int convert(int value, const std::string &elem);
  int unary(const std::string &op, int value);
  int binary(const std::string &op, int lhs, int rhs);
  int compare(const std::string &direction, int lhs, int rhs);
  int select(int pred, int onTrue, int onFalse);
  int reduce(int value, const std::string &op, std::vector<int64_t> axes);
  // [K, M] x [K, N] to [M, N] in f32.
  int dot(int lhs, int rhs);

  // An f32 convert, or the value when it is f32.
  int toF32(int value);
  // To integers: round-to-nearest-even, saturating.
  int toDtype(int value, Dtype dtype);
  // Both values broadcast to their numpy broadcast shape.
  std::pair<int, int> broadcastPair(int lhs, int rhs);

private:
  int emit(const std::string &op, const std::vector<int> &operands, const std::string &attrs,
           Type type, const std::string &region = "");
  int index(int64_t value);
  // Refuses lanes past the slot's partitions.
  void lanes(Slot slot, int64_t first, int64_t count) const;
  // The arena bitcast to a [P, R/K] dtype grid.
  int bitcastTo(Arena arena, Dtype dtype);
  // The grid bitcast back to the arena's bytes.
  void retype(Arena arena, Dtype dtype, int value);
  void reset();

  std::vector<Type> state_;
  std::vector<Type> types_;
  std::ostringstream body_;
  std::vector<int> slots_;
  std::map<std::pair<int, Dtype>, int> bitcasts_;
};

} // namespace nki
