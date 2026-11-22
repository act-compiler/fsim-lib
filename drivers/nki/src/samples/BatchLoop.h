// The batch loop: the trace's one sample, loaded, traced and stored.
#pragma once

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
  // Creates the runtime; the driver outlives the loop.
  BatchLoop(Driver &driver, std::string inputs, std::string outputs);
  BatchLoop(const BatchLoop &) = delete;
  BatchLoop &operator=(const BatchLoop &) = delete;

  // Simulates every sample, then prints the summary line; called once.
  void run();

private:
  // Adds an input allocation.
  void add(const Alloc &alloc);
  // The trace's pipeline, built at the first instruction.
  fsim::Pipeline &upload();
  // Runs the driver's trace, then syncs.
  void trace();
  // Loads the input allocations from their files.
  void load();
  // Stores the sample's outputs, out of the run's clock.
  void store();

  Driver &driver_;
  std::string outputs_;
  Inputs inputs_;
  // The pipeline is destroyed before the runtime.
  fsim::Runtime runtime_;
  std::unique_ptr<fsim::Pipeline> pipeline_;
};

} // namespace nki
