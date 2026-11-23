#include "samples/BatchLoop.h"

#include <chrono>
#include <iostream>
#include <utility>

#include "Driver.h"
#include "samples/Outputs.h"

namespace nki {

namespace {

// Ms a call takes.
template <typename Call> double timed(Call call) {
  auto begin = std::chrono::steady_clock::now();
  call();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

} // namespace

BatchLoop::BatchLoop(Driver &driver, std::string inputs, std::string outputs,
                     const std::string &sampling)
    : driver_(driver), outputs_(std::move(outputs)), inputs_(std::move(inputs), sampling) {
}

void BatchLoop::run() {
  // Above one sample, a trace with a run-time offset runs at a batch size of 1.
  runtimeOffset_ = fsim::Pipeline::num_samples() > 1 && driver_.hasRuntimeOffset();
  trace();
  store();
  // Each later batch traces again with a run-time offset, else replays the first's executables.
  for (first_ = batchSize_; first_ < inputs_.size(); first_ += batchSize_) {
    if (runtimeOffset_) {
      trace();
    } else {
      pipeline_->exclude(timed([&] { load(); }));
      pipeline_->replay();
      pipeline_->sync();
    }
    store();
  }
  pipeline_->summary(std::cout);
  storeDraws(inputs_.draws(), outputs_);
}

void BatchLoop::add(const Alloc &alloc) {
  if (!pipeline_)
    inputs_.add(alloc);
}

fsim::Pipeline &BatchLoop::upload() {
  // A later trace's loads stay out of the run's clock.
  if (pipeline_) {
    pipeline_->exclude(timed([&] { load(); }));
    pipeline_->begin_trace();
    return *pipeline_;
  }
  // B, or 1 with a run-time offset; N draws, more padding the last batch to B.
  batchSize_ = fsim::Pipeline::batch_size(runtime_, driver_.state());
  if (runtimeOffset_)
    batchSize_ = 1;
  inputs_.draw((size_t{fsim::Pipeline::num_samples()} + batchSize_ - 1) / batchSize_ * batchSize_);
  driver_.accelerator().hold(batchSize_);
  load();
  pipeline_ = std::make_unique<fsim::Pipeline>(runtime_, driver_.state(), batchSize_);
  // The run clock starts after the upload.
  pipeline_->begin_run();
  return *pipeline_;
}

void BatchLoop::trace() {
  driver_.trace([&](const Alloc &alloc) { add(alloc); },
                [&]() -> fsim::Pipeline & { return upload(); });
  pipeline_->sync();
}

void BatchLoop::load() {
  for (uint32_t m = 0; m < batchSize_; ++m)
    inputs_.load(driver_.accelerator(), m, first_ + m);
}

void BatchLoop::store() {
  const Accelerator &accelerator = driver_.accelerator();
  pipeline_->exclude(timed([&] {
    // Before the first byte, every sample's outputs must fit.
    if (first_ == 0)
      checkOutputs(accelerator, driver_.outputs(), inputs_.size(), outputs_);
    for (uint32_t m = 0; m < batchSize_; ++m)
      storeOutputs(accelerator, driver_.outputs(), m, first_ + m, outputs_);
  }));
}

} // namespace nki
