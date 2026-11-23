// The pipeline: state tensors on the device, advanced by StableHLO modules.
// Each module's @main takes every state tensor in order and returns them all in order.
// submit() queues a module and returns; modules compile in parallel.
// Modules execute strictly in submission order.
// sync() waits, then copies the state back to host memory; one caller thread drives a pipeline.
// Outside the overlap variant, submit() compiles and executes in place before it returns.
// N samples run in batches of batch_size()'s B; above 1, each state tensor is [B, dims].
// Each sample of a batch starts from its own part of host memory, and sync() hands it back.
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
  std::optional<uint32_t> trigger;     // soft trigger op count, default 10000
  std::optional<uint32_t> threads;     // compile thread count, default 4
  std::optional<uint32_t> num_samples; // samples per run, default 1

  // Consumes --trigger N, --overlap T and --num-samples N from argv and keeps them for every
  // pipeline; returns the new argc. A bad value, or a knob this build lacks,
  // prints the refusal and exits 1.
  static int parse(int argc, char **argv);
};

class Pipeline {
public:
  // Uploads the state of a batch of batch_size samples; runtime and host memory outlive it.
  // Each tensor's host memory holds the batch's samples back to back.
  Pipeline(Runtime &runtime, std::vector<Tensor> state, uint32_t batch_size);
  // Stops the threads; modules not yet run are dropped.
  ~Pipeline();
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  // Whether a graph of ops ops, the state's prologue counted, flushes now.
  bool soft_trigger(size_t ops) const;
  // Queues one module's text; throws the pipeline's error if it has one.
  void submit(std::string module);
  // Runs everything submitted, copies the state to host, rethrows the first error.
  // Above a batch size of 1, one before end_trace throws.
  void sync();
  // A snapshot of the stats; complete after sync().
  Stats stats() const;

  // Starts the run's clock anew, as at construction; called before the first submit.
  void begin_run();
  // Ends the driver's trace of trace_length instructions.
  void end_trace(size_t trace_length);
  // Prints the run's summary line so far, '-' for a value not yet known.
  void summary(std::ostream &out) const;

  // The run: N samples in batches of B, each later batch a replay.
  // N, the --num-samples sample count.
  static uint32_t num_samples();
  // B, the batch size for --num-samples N and one sample's state.
  // Throws when one sample passes the host's or the device's limit.
  static uint32_t batch_size(const Runtime &runtime, const std::vector<Tensor> &state);
  // After a sync(), replays the kept executables in submission order; kept at N above B.
  // Uploads the state anew from the constructor's host memory; sync() reads the samples back.
  void replay();
  // Leaves ms of a driver's loads and stores out of the run's clock.
  void exclude(double ms);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace fsim
