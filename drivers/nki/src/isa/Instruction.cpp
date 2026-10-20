#include "isa/Instruction.h"

#include <map>
#include <stdexcept>

namespace nki {

namespace {

// The kinds of later milestones, refused by name.
const std::map<std::string, std::string> &later() {
  static const std::map<std::string, std::string> kinds = {
      {"tensor_copy", "M2"},          {"tensor_tensor_scan", "M2"}, {"tensor_scalar_arith", "M2"},
      {"affine_select", "M2"},        {"dma_transpose", "M2"},      {"reciprocal", "M2"},
      {"tensor_scalar_reduce", "M2"}, {"select_reduce", "M2"},      {"nc_stream_shuffle", "M2"},
      {"stream_transpose", "M2"},     {"local_gather", "M3"},
  };
  return kinds;
}

} // namespace

// The decode table: a kind to its decode.
Decode decoder(const std::string &kind) {
  static const std::map<std::string, Decode> kinds = {
      {"dma_copy", decodeDmaCopy},
      {"memset", decodeMemset},
      {"activation", decodeActivation},
      {"tensor_tensor_arith", decodeTensorTensorArith},
      {"tensor_reduce_arith", decodeTensorReduceArith},
      {"nc_matmul", decodeNcMatmul},
  };
  auto it = kinds.find(kind);
  if (it != kinds.end())
    return it->second;
  auto next = later().find(kind);
  if (next != later().end())
    throw std::runtime_error("instruction kind '" + kind + "' is milestone " + next->second +
                             "; this build decodes only M1's kinds");
  throw std::runtime_error("unknown instruction kind '" + kind + "'");
}

} // namespace nki
