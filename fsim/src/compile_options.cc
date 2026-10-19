// A protobuf wire encoder for the compile options, varints and
// length-delimited fields only. The field numbers are those of
// xla/pjrt/proto/compile_options.proto.
#include "compile_options.h"

#include <cstdint>
#include <string_view>

namespace fsim {
namespace {

// One message being encoded, field by field.
class ProtoWriter {
public:
  void varint(uint32_t field, uint64_t value) {
    tag(field, 0);
    raw(value);
  }
  void bytes(uint32_t field, std::string_view value) {
    tag(field, 2);
    raw(value.size());
    out_.append(value);
  }
  void message(uint32_t field, const ProtoWriter &value) {
    bytes(field, value.out_);
  }
  const std::string &str() const {
    return out_;
  }

private:
  void tag(uint32_t field, uint32_t wire_type) {
    raw(uint64_t{field} << 3 | wire_type);
  }
  void raw(uint64_t value) {
    for (; value >= 0x80; value >>= 7)
      out_ += static_cast<char>(value | 0x80);
    out_ += static_cast<char>(value);
  }
  std::string out_;
};

} // namespace

std::string compile_options() {
  // ExecutableBuildOptionsProto: field 4, num_replicas; field 5, num_partitions.
  ProtoWriter build;
  build.varint(4, 1);
  build.varint(5, 1);

  // CompileOptionsProto: field 3, executable_build_options.
  ProtoWriter options;
  options.message(3, build);
  return options.str();
}

} // namespace fsim
