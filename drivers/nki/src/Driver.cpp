#include "Driver.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "accelerator/Limits.h"
#include "fsim/pipeline.h"
#include "fsim/runtime.h"
#include "isa/Instruction.h"
#include "trace/Parser.h"

namespace nki {

namespace {

// The state tensors, in @main's order.
std::vector<fsim::Tensor> state(Accelerator &a) {
  using fsim::ElementType;
  return {
      {ElementType::ui8, {1, static_cast<int64_t>(a.hbm.size())}, a.hbm.data()},
      {ElementType::ui8, {kPartitions, kSbufRow}, a.sbuf.data()},
      {ElementType::ui8, {kPartitions, kPsumRow}, a.psum.data()},
      {ElementType::f32, {kPartitions, 1}, a.reduceRegs.data()},
  };
}

// Their types, as the module's parameters.
std::vector<Type> types(const std::vector<fsim::Tensor> &tensors) {
  std::vector<Type> out;
  for (const fsim::Tensor &t : tensors)
    out.push_back({t.type == fsim::ElementType::f32 ? "f32" : "ui8", t.dims});
  return out;
}

} // namespace

Driver::Driver(std::string inputs, std::string outputs, bool validate)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)), validate_(validate) {
}

Driver::~Driver() = default;

void Driver::run(std::istream &in) {
  std::string line;
  size_t number = 0;
  try {
    while (std::getline(in, line)) {
      ++number;
      if (ended_)
        throw std::runtime_error("a record after end");
      if (number == 1) {
        kernel_.header = parseHeader(line);
        if (validate_)
          continue;
        accelerator_ = std::make_unique<Accelerator>(kernel_.header.hbm);
        module_ = std::make_unique<Module>(types(state(*accelerator_)));
      } else if (validate_) {
        check(line);
      } else {
        record(line);
        if (ended_)
          finish();
      }
    }
    if (!ended_)
      throw std::runtime_error("the trace ends without an end record");
  } catch (const std::exception &e) {
    throw std::runtime_error("line " + std::to_string(number) + ": " + e.what());
  }
}

void Driver::record(const std::string &line) {
  Record r = parseRecord(line);
  if (auto *a = std::get_if<Alloc>(&r)) {
    // Inputs load before the upload, at the first step.
    if (!a->input.empty() && pipeline_)
      throw std::runtime_error("input " + a->input + " after the first instruction");
    accelerator_->allocate(*a, inputs_);
  } else if (auto *f = std::get_if<Fields>(&r)) {
    execute(*f);
  } else if (auto *o = std::get_if<Output>(&r)) {
    kernel_.outputs.push_back(std::move(*o));
  } else {
    ended_ = true;
  }
}

void Driver::execute(Fields &fields) {
  Decode decode = decoder(fields.kind());
  resolve(fields);
  InstructionPtr instruction = decode(fields);
  fields.done();
  if (!pipeline_)
    upload();
  if (pipeline_->soft_trigger(module_->ops()))
    submit();
  instruction->semantics(*accelerator_, *module_);
  ++instructionCounter_;
}

void Driver::resolve(Fields &fields) {
  std::vector<View *> views = fields.views();
  // A run-time offset reads state: flush, sync, read back.
  if (std::any_of(views.begin(), views.end(), [](const View *v) { return v->dyn.has_value(); })) {
    if (!pipeline_)
      upload();
    submit();
    sync();
  }
  for (View *v : views)
    *v = accelerator_->resolve(*v);
}

void Driver::check(const std::string &line) {
  ++checked_;
  Record r = parseRecord(line);
  if (auto *f = std::get_if<Fields>(&r)) {
    // Instructions stay undecoded; nc_matmul's views are checked.
    if (f->kind() == "nc_matmul") {
      View dst = f->view("dst");
      if (dst.dyn)
        ++bankUnchecked_;
      checkMatmul(allocs_, dst, f->view("stationary"), f->view("moving"));
    }
  } else if (auto *a = std::get_if<Alloc>(&r)) {
    checkAlloc(allocs_, *a, kernel_.header.hbm);
    allocs_.emplace(a->id, std::move(*a));
  } else if (auto *o = std::get_if<Output>(&r)) {
    if (!allocs_.count(o->id))
      throw std::runtime_error("output " + o->name + " names no allocation %" +
                               std::to_string(o->id));
  } else {
    std::cout << "checked records=" << checked_ << " allocs=" << allocs_.size()
              << " bank_unchecked=" << bankUnchecked_ << "\n";
    ended_ = true;
  }
}

void Driver::upload() {
  runtime_ = std::make_unique<fsim::Runtime>();
  pipeline_ = std::make_unique<fsim::Pipeline>(*runtime_, state(*accelerator_));
  // The run clock starts after the upload.
  pipeline_->begin_run();
}

void Driver::submit() {
  std::string text = module_->text();
  if (!text.empty())
    pipeline_->submit(std::move(text));
}

void Driver::sync() {
  pipeline_->sync();
}

void Driver::finish() {
  if (!pipeline_)
    upload();
  pipeline_->end_trace(instructionCounter_);
  // A run killed in sync keeps this line.
  pipeline_->summary(std::cout);
  submit();
  sync();
  for (const Output &o : kernel_.outputs)
    accelerator_->store(o, outputs_);
  pipeline_->summary(std::cout);
}

} // namespace nki
