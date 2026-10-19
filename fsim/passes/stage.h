// The pass stage over one module, one phase per call, in order:
// parse, stock, library, then serialize or print.
// A phase returns false on failure, with its diagnostics in error().
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

namespace fsim {

class Stage {
public:
  Stage();

  // Parses the module's text.
  bool parse(std::string_view text);
  // Runs CSE, the stock optimization, canonicalize, CSE.
  bool stock();
  // Runs the library's passes, then aliases @main's arguments.
  bool library();
  // Writes the versioned StableHLO artifact.
  bool serialize(std::string &artifact);
  // Writes the module as text.
  void print(std::string &text);

  // The module's op count, the module itself included.
  size_t ops();
  // The module's op names by count, most first.
  std::vector<std::pair<std::string, size_t>> histogram();
  // Makes later phases print each pass's ms and op count.
  void trace() {
    trace_ = true;
  }
  const std::string &error() const {
    return error_;
  }

private:
  // Fails a phase; keeps its name before the diagnostics.
  bool fail(std::string_view phase);
  // Runs pm over the module, traced if asked.
  bool run(mlir::PassManager &pm);

  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::string diagnostics_;
  std::string error_;
  bool trace_ = false;
  mlir::ScopedDiagnosticHandler handler_;
};

} // namespace fsim
