// A protobuf wire encoder for the compile options, varints and
// length-delimited fields only. The field numbers are those of
// xla/pjrt/proto/compile_options.proto.
// It also bakes in each compiler's XLA option overrides.
#include "compile_options.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
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

// One compiler's XLA option overrides, as name=value.
struct CompilerFlags {
  std::string_view compiler;
  std::array<std::string_view, 2> flags;
};

// Every compiler's overrides; an empty flag is unused.
constexpr std::array<CompilerFlags, 6> kCompilerFlags{{
    {"xla-0.11-cpu",
     {"xla_backend_optimization_level=1", "xla_cpu_copy_insertion_use_region_analysis=True"}},
    {"xla-0.4-cpu", {"xla_backend_optimization_level=1"}},
    {"custom-cpu",
     {"xla_backend_optimization_level=1", "xla_cpu_copy_insertion_use_region_analysis=True"}},
    {"xla-0.11-gpu", {"xla_gpu_autotune_level=0", "xla_gpu_enable_triton_gemm=False"}},
    {"xla-0.4-gpu", {"xla_gpu_autotune_level=0", "xla_gpu_enable_triton_gemm=False"}},
    {"custom-gpu", {"xla_gpu_autotune_level=0", "xla_gpu_enable_triton_gemm=False"}},
}};

// The build's compiler row; a missing row throws.
const CompilerFlags &compiler_flags() {
  for (const CompilerFlags &row : kCompilerFlags)
    if (row.compiler == std::string_view{FSIM_COMPILER_ID})
      return row;
  throw std::logic_error(std::string{"no XLA option overrides for "} + FSIM_COMPILER_ID);
}

// One env_option_overrides entry: field 7, a map entry of key 1 and
// value 2, an OptionOverrideProto whose field 1 is string_field.
void override_option(ProtoWriter &options, std::string_view flag) {
  size_t equals = flag.find('=');
  ProtoWriter value;
  value.bytes(1, flag.substr(equals + 1));
  ProtoWriter entry;
  entry.bytes(1, flag.substr(0, equals));
  entry.message(2, value);
  options.message(7, entry);
}

} // namespace

std::string compile_options() {
  // ExecutableBuildOptionsProto: field 4, num_replicas; field 5, num_partitions.
  ProtoWriter build;
  build.varint(4, 1);
  build.varint(5, 1);

  // CompileOptionsProto: field 3, executable_build_options.
  ProtoWriter options;
  options.message(3, build);

  // The compiler's overrides.
  for (std::string_view flag : compiler_flags().flags)
    if (!flag.empty())
      override_option(options, flag);
  return options.str();
}

} // namespace fsim
