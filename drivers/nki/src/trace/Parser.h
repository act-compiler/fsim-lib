// One line to one record: tokenizer and view parser.
// An instruction record stays as its fields, not yet decoded.
#pragma once

#include <string>
#include <variant>

#include "trace/Fields.h"
#include "trace/Kernel.h"

namespace nki {

struct End {};

using Record = std::variant<Alloc, Fields, Output, End>;

// The header line; throws on anything else.
Header parseHeader(const std::string &line);
// Any later line; throws on a bad or reserved record.
Record parseRecord(const std::string &line);

} // namespace nki
