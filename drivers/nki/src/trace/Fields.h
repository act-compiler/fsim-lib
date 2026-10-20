// An instruction record's fields, consumed by name by its decode.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "trace/View.h"

namespace nki {

// A number as the record spells it.
struct Number {
  double value = 0;
  std::optional<int64_t> integer;
};
struct Word {
  std::string text;
};
struct None {};
struct Affine {
  std::vector<Entry> pattern;
  int64_t offset = 0;
};
using List = std::vector<Number>;
using Value = std::variant<None, bool, Number, Word, List, Affine, View>;

// A record's fields; getters consume, done() refuses the rest.
class Fields {
public:
  Fields(std::string kind, std::map<std::string, Value> fields);

  const std::string &kind() const {
    return kind_;
  }
  // Every view field not yet taken.
  std::vector<View *> views();
  View view(const std::string &name);
  std::optional<View> optView(const std::string &name);
  Number number(const std::string &name);
  bool flag(const std::string &name);
  std::optional<bool> optFlag(const std::string &name);
  std::string word(const std::string &name);
  std::optional<std::string> optWord(const std::string &name);
  // A number, a view or none.
  Value operand(const std::string &name);
  // Takes the field whatever its value.
  void skip(const std::string &name);
  // Refuses any field no getter took.
  void done() const;
  // Throws naming the kind and the field.
  [[noreturn]] void fail(const std::string &name, const std::string &what) const;

private:
  Value take(const std::string &name);
  std::string kind_;
  std::map<std::string, Value> fields_;
};

} // namespace nki
