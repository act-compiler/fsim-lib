// The driver over a trace: trace() runs its instructions as they arrive,
// through the pipeline its caller builds, and ends the trace at end.
// A parse-only reading finds a run-time offset before the caller picks its batch size.
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
  // Reads trace, a file or "-" for stdin, on each reading; stdin is read once.
  explicit Driver(std::string trace);
  ~Driver();

  // Parses and checks every record; throws naming the line.
  void validate();
  // Whether an instruction takes a +s() offset; parses only, throws naming the line.
  bool hasRuntimeOffset();
  // One reading through upload's pipeline, taken at the first instruction; add takes each
  // input allocation. Throws naming the line.
  void trace(std::function<void(const Alloc &)> add, std::function<fsim::Pipeline &()> upload);

  // The host copies, after a trace's header.
  Accelerator &accelerator();
  // The state tensors over them, in @main's order, one sample's dims each;
  // the oob flag last when the trace holds a +s() offset.
  std::vector<fsim::Tensor> state();
  // The last reading's outputs.
  const std::vector<Output> &outputs() const;

private:
  // The trace from its first line: the file anew, or stdin's text.
  std::unique_ptr<std::istream> open() const;
  // Every line, the header first, then start if any, then each through each;
  // throws naming the line.
  void read(void (Driver::*each)(const std::string &), void (Driver::*start)() = nullptr);
  // A trace's start, after the header: afresh but for the host copies.
  void begin();
  // One line past the header.
  void record(const std::string &line);
  // One line past the header: parsed, its memory checked.
  void check(const std::string &line);
  // One line past the header: parsed, its offsets noted.
  void scan(const std::string &line);
  // Decodes and runs one instruction, its views resolved.
  void execute(Fields &fields);
  // Checks every view, bounding its run-time offset.
  void resolve(Fields &fields);

  // Takes the trace's pipeline, which upload_ gives.
  void upload();
  // Submits the module, if it holds ops.
  void submit();
  // End the trace, fsim's summary, the last submit.
  void finish();

  std::string path_, stdin_;
  Kernel kernel_;
  // The checked allocations, validating.
  std::map<int64_t, Alloc> allocs_;
  size_t checked_ = 0;
  // nc_matmul dsts whose bank a run-time offset hides.
  size_t bankUnchecked_ = 0;
  // An instruction takes a +s() offset, scanning.
  bool runtimeOffset_ = false;
  // A +s() offset in the trace: the state holds the oob flag.
  bool oob_ = false;
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
