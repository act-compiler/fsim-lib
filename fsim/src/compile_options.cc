// A protobuf wire encoder for the compile options, varints and
// length-delimited fields only. Field numbers cite XLA's protos at
// commit 6b73c4c49eb0c5aee1963c1e34c7cc8ba92ac202, by file and line.
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
  // ExecutableBuildOptionsProto, xla/pjrt/proto/compile_options.proto:26.
  ProtoWriter build;
  // num_replicas = 4, xla/pjrt/proto/compile_options.proto:51.
  build.varint(4, 1);
  // num_partitions = 5, xla/pjrt/proto/compile_options.proto:54.
  build.varint(5, 1);

  // CompileOptionsProto, xla/pjrt/proto/compile_options.proto:175.
  ProtoWriter options;
  // executable_build_options = 3, xla/pjrt/proto/compile_options.proto:179.
  options.message(3, build);
  return options.str();
}

} // namespace fsim
