#include "trace/Parser.h"

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string_view>

namespace nki {

namespace {

[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error(what);
}

// A record's fields: single spaces, no empty field.
std::vector<std::string> tokens(const std::string &line) {
  std::vector<std::string> out;
  size_t at = 0;
  while (true) {
    size_t end = line.find(' ', at);
    std::string token = line.substr(at, end == std::string::npos ? end : end - at);
    if (token.empty())
      fail("empty field in '" + line + "'");
    out.push_back(std::move(token));
    if (end == std::string::npos)
      return out;
    at = end + 1;
  }
}

// A cursor over one field's text.
class Cursor {
public:
  explicit Cursor(std::string_view text) : text_(text), rest_(text) {
  }

  bool done() const {
    return rest_.empty();
  }
  bool eat(std::string_view prefix) {
    if (!rest_.starts_with(prefix))
      return false;
    rest_.remove_prefix(prefix.size());
    return true;
  }
  void expect(std::string_view prefix) {
    if (!eat(prefix))
      fail("expected '" + std::string(prefix) + "'");
  }
  int64_t integer() {
    size_t n = rest_.starts_with('-') ? 1 : 0;
    size_t digits = n;
    while (digits < rest_.size() && std::isdigit(static_cast<unsigned char>(rest_[digits])))
      ++digits;
    if (digits == n)
      fail("expected an integer");
    int64_t value = std::stoll(std::string(rest_.substr(0, digits)));
    rest_.remove_prefix(digits);
    return value;
  }
  // The text up to any of the stop characters.
  std::string_view until(std::string_view stops) {
    size_t n = rest_.find_first_of(stops);
    std::string_view word = rest_.substr(0, n);
    rest_.remove_prefix(word.size());
    return word;
  }
  void end() {
    if (!done())
      fail("trailing '" + std::string(rest_) + "'");
  }
  [[noreturn]] void fail(const std::string &what) const {
    nki::fail(what + " at '" + std::string(rest_) + "' in '" + std::string(text_) + "'");
  }

private:
  std::string_view text_;
  std::string_view rest_;
};

Arena arena(Cursor &c) {
  if (c.eat("hbm") || c.eat("private_hbm") || c.eat("shared_hbm"))
    return Arena::hbm;
  if (c.eat("sbuf"))
    return Arena::sbuf;
  if (c.eat("psum"))
    return Arena::psum;
  c.fail("expected an arena");
}

Dtype dtype(Cursor &c) {
  std::string_view name = c.until("[");
  static const std::map<std::string_view, Dtype> dtypes = {
      {"bf16", Dtype::bf16}, {"f16", Dtype::f16}, {"f32", Dtype::f32}, {"i32", Dtype::i32},
      {"u8", Dtype::u8},     {"u16", Dtype::u16}, {"u32", Dtype::u32}};
  auto it = dtypes.find(name);
  if (it == dtypes.end())
    c.fail("unknown dtype '" + std::string(name) + "'");
  return it->second;
}

// addr := bytes | partition "," bytes
void address(Cursor &c, Arena a, int64_t &partition, int64_t &byte) {
  partition = 0;
  if (a != Arena::hbm) {
    partition = c.integer();
    c.expect(",");
  }
  byte = c.integer();
}

// pattern := "[" "[" stride "," count "]" ("," ...)* "]"
std::vector<Entry> pattern(Cursor &c) {
  std::vector<Entry> entries;
  c.expect("[");
  do {
    Entry e;
    c.expect("[");
    e.stride = c.integer();
    c.expect(",");
    e.count = c.integer();
    c.expect("]");
    entries.push_back(e);
  } while (c.eat(","));
  c.expect("]");
  return entries;
}

// view := arena "@" addr ":" dtype pattern dyn?
View view(Cursor &c) {
  View v;
  v.arena = arena(c);
  c.expect("@");
  address(c, v.arena, v.partition, v.byte);
  c.expect(":");
  v.dtype = dtype(c);
  v.pattern = pattern(c);
  // dyn := "+s(" view ")*" int
  if (c.eat("+s(")) {
    Dyn d;
    d.view = std::make_shared<const View>(view(c));
    c.expect(")*");
    d.scale = c.integer();
    v.dyn = std::move(d);
  } else if (c.eat("+r")) {
    c.fail("a register offset is reserved until M3");
  }
  return v;
}

bool isView(std::string_view text) {
  for (std::string_view a : {"hbm@", "sbuf@", "psum@", "private_hbm@", "shared_hbm@"})
    if (text.starts_with(a))
      return true;
  return false;
}

// A Python int or float repr.
Number number(const std::string &text) {
  Number n;
  char *end = nullptr;
  n.value = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size())
    fail("bad number '" + text + "'");
  if (text.find_first_not_of("-0123456789") == std::string::npos)
    n.integer = std::stoll(text);
  return n;
}

bool isNumber(const std::string &text) {
  return std::isdigit(static_cast<unsigned char>(text[0])) || text[0] == '-' || text[0] == '.' ||
         text == "inf" || text == "nan";
}

Value value(const std::string &text) {
  if (text == "true")
    return true;
  if (text == "false")
    return false;
  if (text == "none")
    return None{};
  if (isView(text)) {
    Cursor c(text);
    View v = view(c);
    c.end();
    return v;
  }
  // list := "[" (number ("," number)*)? "]"
  if (text.starts_with("[")) {
    if (text.back() != ']')
      fail("bad list '" + text + "'");
    List list;
    std::string body = text.substr(1, text.size() - 2);
    for (size_t at = 0; !body.empty() && at <= body.size();) {
      size_t end = body.find(',', at);
      list.push_back(number(body.substr(at, end == std::string::npos ? end : end - at)));
      at = end == std::string::npos ? body.size() + 1 : end + 1;
    }
    return list;
  }
  // affine := "aff(" pattern "," int ")"
  if (text.starts_with("aff(")) {
    Cursor c(text);
    c.expect("aff(");
    Affine a;
    a.pattern = pattern(c);
    c.expect(",");
    a.offset = c.integer();
    c.expect(")");
    c.end();
    return a;
  }
  if (isNumber(text))
    return number(text);
  if (text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
      std::string::npos)
    fail("bad value '" + text + "'");
  return Word{text};
}

// id := "%" int
int64_t id(const std::string &text) {
  Cursor c(text);
  c.expect("%");
  int64_t n = c.integer();
  c.end();
  return n;
}

int64_t integer(const std::string &text) {
  Cursor c(text);
  int64_t n = c.integer();
  c.end();
  return n;
}

// alloc := "alloc" id arena "@" addr dtype "[" dims "]" placed name? input?
Alloc alloc(const std::vector<std::string> &t) {
  if (t.size() < 5 || t.size() > 7)
    fail("alloc takes 4 to 6 fields");
  Alloc a;
  a.id = id(t[1]);
  Cursor at(t[2]);
  a.arena = arena(at);
  at.expect("@");
  address(at, a.arena, a.partition, a.byte);
  at.end();
  Cursor type(t[3]);
  a.dtype = dtype(type);
  type.expect("[");
  do
    a.dims.push_back(type.integer());
  while (type.eat(","));
  type.expect("]");
  type.end();
  if (t[4] != "kernel" && t[4] != "driver")
    fail("placed is kernel or driver, not '" + t[4] + "'");
  a.kernel = t[4] == "kernel";
  size_t i = 5;
  if (i < t.size() && t[i].starts_with("name="))
    a.name = t[i++].substr(5);
  if (i < t.size() && t[i].starts_with("input="))
    a.input = t[i++].substr(6);
  if (i != t.size())
    fail("unknown alloc field '" + t[i] + "'");
  return a;
}

// instr := kind (field "=" value)*
Fields fields(const std::vector<std::string> &t) {
  const std::string &kind = t[0];
  std::map<std::string, Value> named;
  for (size_t i = 1; i < t.size(); ++i) {
    size_t eq = t[i].find('=');
    if (eq == std::string::npos || eq == 0)
      fail(kind + ": field '" + t[i] + "' is not name=value");
    std::string name = t[i].substr(0, eq);
    if (!named.emplace(name, value(t[i].substr(eq + 1))).second)
      fail(kind + ": field '" + name + "' given twice");
  }
  return Fields(kind, std::move(named));
}

} // namespace

Header parseHeader(const std::string &line) {
  std::vector<std::string> t = tokens(line);
  if (t.size() != 5 || t[0] != "nki-trace")
    fail("the first line is not a trace header: '" + line + "'");
  Header h;
  h.version = integer(t[1]);
  if (h.version != 1)
    fail("trace format version " + t[1] + "; this build reads version 1");
  if (t[2] != "target=trn2" || t[3] != "lnc=1")
    fail("the target is '" + t[2] + " " + t[3] + "'; this build simulates target=trn2 lnc=1");
  if (!t[4].starts_with("hbm="))
    fail("expected hbm=<bytes>, got '" + t[4] + "'");
  h.hbm = integer(t[4].substr(4));
  return h;
}

Record parseRecord(const std::string &line) {
  std::vector<std::string> t = tokens(line);
  const std::string &head = t[0];
  if (head == "alloc")
    return alloc(t);
  if (head == "output") {
    if (t.size() != 3)
      fail("output takes a name and an id");
    return Output{t[1], id(t[2])};
  }
  if (head == "end") {
    if (t.size() != 1)
      fail("end takes no fields");
    return End{};
  }
  // The control records stay reserved.
  for (const char *control : {"reg.load", "reg.set", "loop", "endloop"})
    if (head == control)
      fail("record '" + head + "' is reserved until M3");
  return fields(t);
}

Fields::Fields(std::string kind, std::map<std::string, Value> fields)
    : kind_(std::move(kind)), fields_(std::move(fields)) {
}

std::vector<View *> Fields::views() {
  std::vector<View *> out;
  for (auto &[name, value] : fields_)
    if (auto *v = std::get_if<View>(&value))
      out.push_back(v);
  return out;
}

void Fields::fail(const std::string &name, const std::string &what) const {
  nki::fail(kind_ + ": field '" + name + "' " + what);
}

Value Fields::take(const std::string &name) {
  auto it = fields_.find(name);
  if (it == fields_.end())
    fail(name, "is missing");
  Value v = std::move(it->second);
  fields_.erase(it);
  return v;
}

View Fields::view(const std::string &name) {
  Value v = take(name);
  if (!std::holds_alternative<View>(v))
    fail(name, "is not a view");
  return std::get<View>(std::move(v));
}

std::optional<View> Fields::optView(const std::string &name) {
  Value v = take(name);
  if (std::holds_alternative<None>(v))
    return std::nullopt;
  if (!std::holds_alternative<View>(v))
    fail(name, "is neither a view nor none");
  return std::get<View>(std::move(v));
}

Number Fields::number(const std::string &name) {
  Value v = take(name);
  if (!std::holds_alternative<Number>(v))
    fail(name, "is not a number");
  return std::get<Number>(v);
}

bool Fields::flag(const std::string &name) {
  Value v = take(name);
  if (!std::holds_alternative<bool>(v))
    fail(name, "is not true or false");
  return std::get<bool>(v);
}

std::optional<bool> Fields::optFlag(const std::string &name) {
  Value v = take(name);
  if (std::holds_alternative<None>(v))
    return std::nullopt;
  if (!std::holds_alternative<bool>(v))
    fail(name, "is not true, false or none");
  return std::get<bool>(v);
}

std::string Fields::word(const std::string &name) {
  Value v = take(name);
  if (!std::holds_alternative<Word>(v))
    fail(name, "is not a name");
  return std::get<Word>(v).text;
}

std::optional<std::string> Fields::optWord(const std::string &name) {
  Value v = take(name);
  if (std::holds_alternative<None>(v))
    return std::nullopt;
  if (!std::holds_alternative<Word>(v))
    fail(name, "is neither a name nor none");
  return std::get<Word>(v).text;
}

Value Fields::operand(const std::string &name) {
  Value v = take(name);
  if (!std::holds_alternative<None>(v) && !std::holds_alternative<Number>(v) &&
      !std::holds_alternative<View>(v))
    fail(name, "is not a number, a view or none");
  return v;
}

void Fields::skip(const std::string &name) {
  take(name);
}

void Fields::done() const {
  if (!fields_.empty())
    nki::fail(kind_ + ": unknown field '" + fields_.begin()->first + "'");
}

} // namespace nki
