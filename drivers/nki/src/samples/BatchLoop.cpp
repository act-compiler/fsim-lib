#include "samples/BatchLoop.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
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

BatchLoop::BatchLoop(Driver &driver, std::string inputs, std::string outputs)
    : driver_(driver), outputs_(std::move(outputs)), inputs_(std::move(inputs)) {
}

void BatchLoop::run() {
  trace();
  store();
  pipeline_->summary(std::cout);
}

void BatchLoop::add(const Alloc &alloc) {
  inputs_.add(alloc);
}

fsim::Pipeline &BatchLoop::upload() {
  // B; N draws.
  batchSize_ = fsim::Pipeline::batch_size(runtime_, driver_.state());
  // One batch holds every sample.
  if (batchSize_ < fsim::Pipeline::num_samples())
    throw std::runtime_error("--num-samples " + std::to_string(fsim::Pipeline::num_samples()) +
                             " passes one batch of " + std::to_string(batchSize_) + " samples");
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
    for (uint32_t m = 0; m < batchSize_; ++m)
      storeOutputs(accelerator, driver_.outputs(), m, first_ + m, outputs_);
  }));
}

} // namespace nki
