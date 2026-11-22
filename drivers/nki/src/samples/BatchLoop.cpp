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
  load();
  pipeline_ = std::make_unique<fsim::Pipeline>(runtime_, driver_.state());
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
  inputs_.load(driver_.accelerator());
}

void BatchLoop::store() {
  const Accelerator &accelerator = driver_.accelerator();
  pipeline_->exclude(timed([&] { storeOutputs(accelerator, driver_.outputs(), outputs_); }));
}

} // namespace nki
