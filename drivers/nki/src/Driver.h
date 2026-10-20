// The driver over a trace: instructions run as they arrive,
// through fsim's Runtime and Pipeline.
// After end: the outputs and fsim's summary.
// Validating, it parses and checks records and runs nothing.
#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>

#include "accelerator/Accelerator.h"
#include "stablehlo/Module.h"
#include "trace/Fields.h"
#include "trace/Kernel.h"

namespace fsim {
class Runtime;
class Pipeline;
} // namespace fsim

namespace nki {

class Driver {
public:
  Driver(std::string inputs, std::string outputs, bool validate = false);
  ~Driver();

  // Reads, runs and ends one trace; throws naming the line.
  void run(std::istream &in);

private:
  // One line past the header.
  void record(const std::string &line);
  // One line past the header: parsed, its memory checked.
  void check(const std::string &line);
  // Decodes and runs one instruction, its views resolved.
  void execute(Fields &fields);
  // Substitutes run-time offsets, then checks every view.
  void resolve(Fields &fields);

  // Uploads the state; the host copies outlive the pipeline.
  void upload();
  // Submits the module, if it holds ops.
  void submit();
  // Runs everything submitted, copies state back.
  void sync();
  // Submit, sync, outputs, fsim's summary.
  void finish();

  std::string inputs_, outputs_;
  bool validate_;
  Kernel kernel_;
  // The checked allocations, validating.
  std::map<int64_t, Alloc> allocs_;
  size_t checked_ = 0;
  // nc_matmul dsts whose bank a run-time offset hides.
  size_t bankUnchecked_ = 0;
  // Destroyed last: the pipeline holds its host copies.
  std::unique_ptr<Accelerator> accelerator_;
  std::unique_ptr<fsim::Runtime> runtime_;
  std::unique_ptr<fsim::Pipeline> pipeline_;
  std::unique_ptr<Module> module_;
  // Instructions traced.
  size_t instructionCounter_ = 0;
  bool ended_ = false;
};

} // namespace nki
