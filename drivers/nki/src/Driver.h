// The driver over a trace: trace() runs its instructions as they arrive,
// through the pipeline its caller builds, and ends the trace at end.
// Validating, it parses and checks records and runs nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "accelerator/Accelerator.h"
#include "fsim/tensor.h"
#include "stablehlo/Module.h"
#include "trace/Fields.h"
#include "trace/Kernel.h"

namespace fsim {
class Pipeline;
} // namespace fsim

namespace nki {

class Driver {
public:
  // Reads trace, a file or "-" for stdin.
  explicit Driver(std::string trace);
  ~Driver();

  // Parses and checks every record; throws naming the line.
  void validate();
  // One reading through upload's pipeline, taken at the first instruction; add takes each
  // input allocation. Throws naming the line.
  void trace(std::function<void(const Alloc &)> add, std::function<fsim::Pipeline &()> upload);

  // The host copies, after a trace's header.
  Accelerator &accelerator();
  // The state tensors over them, in @main's order, one sample's dims each.
  std::vector<fsim::Tensor> state();
  // The last reading's outputs.
  const std::vector<Output> &outputs() const;

private:
  // The trace from its first line: the file or stdin.
  std::unique_ptr<std::istream> open() const;
  // Every line, the header first, then start if any, then each through each;
  // throws naming the line.
  void read(void (Driver::*each)(const std::string &), void (Driver::*start)() = nullptr);
  // A trace's start, after the header: the host copies and the module.
  void begin();
  // One line past the header.
  void record(const std::string &line);
  // One line past the header: parsed, its memory checked.
  void check(const std::string &line);
  // Decodes and runs one instruction, its views resolved.
  void execute(Fields &fields);
  // Substitutes run-time offsets, then checks every view.
  void resolve(Fields &fields);

  // Takes the trace's pipeline, which upload_ builds.
  void upload();
  // Submits the module, if it holds ops.
  void submit();
  // Runs everything submitted, copies state back.
  void sync();
  // End the trace, fsim's summary, the last submit.
  void finish();

  std::string path_;
  Kernel kernel_;
  // The checked allocations, validating.
  std::map<int64_t, Alloc> allocs_;
  size_t checked_ = 0;
  // nc_matmul dsts whose bank a run-time offset hides.
  size_t bankUnchecked_ = 0;
  // Its host copies are the pipeline's state.
  std::unique_ptr<Accelerator> accelerator_;
  // The trace's callbacks, and its pipeline once built.
  std::function<void(const Alloc &)> add_;
  std::function<fsim::Pipeline &()> upload_;
  fsim::Pipeline *pipeline_ = nullptr;
  std::unique_ptr<Module> module_;
  // Instructions traced.
  size_t instructionCounter_ = 0;
  bool ended_ = false;
};

} // namespace nki
