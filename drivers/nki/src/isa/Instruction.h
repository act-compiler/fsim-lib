// An instruction: one record's decoded fields and its semantics.
// Its views arrive resolved: no run-time offset, each checked.
#pragma once

#include <memory>
#include <string>

#include "trace/Fields.h"

namespace nki {

class Accelerator;
class Module;

class Instruction {
public:
  virtual ~Instruction() = default;
  // Reads the views, computes, writes the destination.
  virtual void semantics(Accelerator &accelerator, Module &module) const = 0;
};

using InstructionPtr = std::unique_ptr<const Instruction>;
using Decode = InstructionPtr (*)(Fields &);

// The kind's decode; throws for an undecoded kind.
Decode decoder(const std::string &kind);

// One decode per M1 kind; each refuses unknown fields.
InstructionPtr decodeDmaCopy(Fields &fields);
InstructionPtr decodeMemset(Fields &fields);
InstructionPtr decodeActivation(Fields &fields);
InstructionPtr decodeTensorTensorArith(Fields &fields);
InstructionPtr decodeTensorReduceArith(Fields &fields);
InstructionPtr decodeNcMatmul(Fields &fields);

} // namespace nki
