// The pipeline: state tensors on the device, advanced by StableHLO modules.
// Each module's @main takes every state tensor in order and returns them all in order.
// submit() queues a module and returns; modules compile in parallel.
// Modules execute strictly in submission order.
// sync() waits, then copies the state back to host memory; one caller thread drives a pipeline.
// Outside the overlap variant, submit() compiles and executes in place before it returns.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fsim/runtime.h"
#include "fsim/tensor.h"

namespace fsim {

// One module's timings, in ms.
struct ModuleStats {
  double compile_ms = 0;
  double execute_ms = 0;
};

// The overlap executor's idle time, in ms.
struct Waits {
  double tracing_ms = 0; // next module not yet submitted
  double compile_ms = 0; // next module submitted, not compiled
};

// Every submitted module's stats, in submission order.
struct Stats {
  std::vector<ModuleStats> modules;
  std::optional<Waits> waits; // set in the overlap variant only

  size_t module_count() const {
    return modules.size();
  }
};

// Run-time knobs; an unset one takes the library's default.
struct Knobs {
  std::optional<uint32_t> trigger; // soft trigger op count, default 10000
  std::optional<uint32_t> threads; // compile thread count, default 4

  // Consumes --trigger N and --overlap T from argv and keeps them for every
  // pipeline; returns the new argc. A bad value, or a knob this build lacks,
  // prints the refusal and exits 1.
  static int parse(int argc, char **argv);
};

class Pipeline {
public:
  // Uploads the state; runtime and host memory must outlive the pipeline.
  Pipeline(Runtime &runtime, std::vector<Tensor> state);
  // Stops the threads; modules not yet run are dropped.
  ~Pipeline();
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  // Whether a graph of ops ops, the state's prologue counted, flushes now.
  bool soft_trigger(size_t ops) const;
  // Queues one module's text; throws the pipeline's error if it has one.
  void submit(std::string module);
  // Runs everything submitted, copies the state to host, rethrows the first error.
  void sync();
  // A snapshot of the stats; complete after sync().
  Stats stats() const;

  // Restarts the run's clock, started at construction.
  void begin_run();
  // Ends the driver's trace of trace_length instructions.
  void end_trace(size_t trace_length);
  // Prints the run's summary line so far, '-' for a value not yet known.
  void summary(std::ostream &out) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace fsim
