// The batch loop: --num-samples N samples over the case's input sets, in batches of B.
// The first batch traces into a pipeline at batch size B; later batches replay its executables.
// The last summary line covers every sample.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "fsim/pipeline.h"
#include "fsim/runtime.h"
#include "samples/Inputs.h"
#include "trace/Kernel.h"

namespace nki {

class Driver;

class BatchLoop {
public:
  // Creates the runtime; the driver outlives the loop. sampling is Inputs'.
  BatchLoop(Driver &driver, std::string inputs, std::string outputs, const std::string &sampling);
  BatchLoop(const BatchLoop &) = delete;
  BatchLoop &operator=(const BatchLoop &) = delete;

  // Simulates every sample, prints the summary line, then writes draws.bin; called once.
  void run();

private:
  // Adds an input allocation.
  void add(const Alloc &alloc);
  // The trace's pipeline, built at the first instruction.
  fsim::Pipeline &upload();
  // Runs the driver's trace over the batch, then syncs.
  void trace();
  // Loads every sample of the batch from its input set.
  void load();
  // Stores every sample of the batch, out of the run's clock.
  void store();

  Driver &driver_;
  std::string outputs_;
  Inputs inputs_;
  uint32_t batchSize_ = 1; // B
  size_t first_ = 0;       // the batch's first sample
  // The pipeline is destroyed before the runtime.
  fsim::Runtime runtime_;
  std::unique_ptr<fsim::Pipeline> pipeline_;
};

} // namespace nki
