// The pipeline: state tensors on the device, advanced by StableHLO modules.
// Each module's @main takes every state tensor in order and returns them all
// in order. submit() queues a module and returns; modules compile in parallel
// and execute strictly in submission order. sync() waits, then copies the
// state back to host memory. One caller thread drives a pipeline.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fsim/runtime.h"
#include "fsim/tensor.h"

namespace fsim {

// One module's timings, in ms since the pipeline's construction.
struct ModuleStats {
  size_t bytes = 0;
  double compile_begin_ms = 0;
  double compile_end_ms = 0;
  double execute_begin_ms = 0;
  double execute_end_ms = 0;

  double compile_ms() const {
    return compile_end_ms - compile_begin_ms;
  }
  double execute_ms() const {
    return execute_end_ms - execute_begin_ms;
  }
};

// Every submitted module's stats, in submission order.
struct Stats {
  std::vector<ModuleStats> modules;

  size_t module_count() const {
    return modules.size();
  }
  size_t total_bytes() const;
};

class Pipeline {
public:
  // Uploads the state; runtime and host memory must outlive the pipeline.
  Pipeline(Runtime &runtime, std::vector<Tensor> state);
  // Stops the threads; modules not yet run are dropped.
  ~Pipeline();
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  // Queues one module's text; throws the pipeline's error if it has one.
  void submit(std::string module);
  // Runs everything submitted, copies the state to host, rethrows the first error.
  void sync();
  // A snapshot of the stats; complete after sync().
  Stats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace fsim
