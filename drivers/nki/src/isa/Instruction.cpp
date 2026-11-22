#include "isa/Instruction.h"

#include <map>
#include <stdexcept>

namespace nki {

// The decode table: a kind to its decode.
Decode decoder(const std::string &kind) {
  static const std::map<std::string, Decode> kinds = {
      {"dma_copy", decodeDmaCopy},
      {"memset", decodeMemset},
      {"activation", decodeActivation},
      {"tensor_tensor_arith", decodeTensorTensorArith},
      {"tensor_reduce_arith", decodeTensorReduceArith},
      {"nc_matmul", decodeNcMatmul},
      {"affine_select", decodeAffineSelect},
      {"tensor_copy", decodeTensorCopy},
      {"tensor_scalar_arith", decodeTensorScalarArith},
  };
  auto it = kinds.find(kind);
  if (it != kinds.end())
    return it->second;
  throw std::runtime_error("unknown instruction kind '" + kind + "'");
}

} // namespace nki
