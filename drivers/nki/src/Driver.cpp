#include "Driver.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "accelerator/Limits.h"
#include "fsim/pipeline.h"
#include "isa/Instruction.h"
#include "trace/Parser.h"

namespace nki {

namespace {

// Their types, as the module's parameters.
std::vector<Type> types(const std::vector<fsim::Tensor> &state) {
  std::vector<Type> out;
  for (const fsim::Tensor &t : state)
    out.push_back({t.type == fsim::ElementType::f32 ? "f32" : "ui8", t.dims});
  return out;
}

} // namespace

Driver::Driver(std::string trace) : path_(std::move(trace)) {
  // Stdin is read once, for every reading.
  if (path_ == "-")
    stdin_.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
}

Driver::~Driver() = default;

void Driver::validate() {
  read(&Driver::check);
}

void Driver::trace(std::function<void(const Alloc &)> add,
                   std::function<fsim::Pipeline &()> upload) {
  add_ = std::move(add);
  upload_ = std::move(upload);
  // A +s() offset anywhere keeps the oob flag in the state.
  std::unique_ptr<std::istream> in = open();
  std::string line;
  while (!oob_ && std::getline(*in, line))
    oob_ = line.find("+s(") != std::string::npos;
  read(&Driver::record, &Driver::begin);
}

Accelerator &Driver::accelerator() {
  return *accelerator_;
}

std::vector<fsim::Tensor> Driver::state() {
  using fsim::ElementType;
  Accelerator &a = *accelerator_;
  std::vector<fsim::Tensor> state = {
      {ElementType::ui8, {1, a.rowBytes(Arena::hbm)}, a.hbm.data()},
      {ElementType::ui8, {kPartitions, kSbufRow}, a.sbuf.data()},
      {ElementType::ui8, {kPartitions, kPsumRow}, a.psum.data()},
      {ElementType::f32, {kPartitions, 1}, a.reduceRegs.data()},
      {ElementType::f32, {kPartitions, 1}, a.accumulatorRegs.data()},
  };
  if (oob_)
    state.push_back({ElementType::ui8, {1}, a.oob.data()});
  return state;
}

const std::vector<Output> &Driver::outputs() const {
  return kernel_.outputs;
}

std::unique_ptr<std::istream> Driver::open() const {
  if (path_ == "-")
    return std::make_unique<std::istringstream>(stdin_);
  auto in = std::make_unique<std::ifstream>(path_);
  if (!*in)
    throw std::runtime_error("cannot read " + path_);
  return in;
}

void Driver::read(void (Driver::*each)(const std::string &), void (Driver::*start)()) {
  std::unique_ptr<std::istream> in = open();
  std::string line;
  size_t number = 0;
  try {
    while (std::getline(*in, line)) {
      ++number;
      if (ended_)
        throw std::runtime_error("a record after end");
      if (number == 1) {
        kernel_.header = parseHeader(line);
        if (start)
          (this->*start)();
      } else {
        (this->*each)(line);
      }
    }
    if (!ended_)
      throw std::runtime_error("the trace ends without an end record");
  } catch (const std::exception &e) {
    throw std::runtime_error("line " + std::to_string(number) + ": " + e.what());
  }
}

void Driver::begin() {
  accelerator_ = std::make_unique<Accelerator>(kernel_.header.hbm);
  module_ = std::make_unique<Module>(types(state()));
}

void Driver::record(const std::string &line) {
  Record r = parseRecord(line);
  if (auto *a = std::get_if<Alloc>(&r)) {
    // Inputs load before the upload, at the first step.
    if (!a->input.empty() && pipeline_)
      throw std::runtime_error("input " + a->input + " after the first instruction");
    accelerator_->allocate(*a);
    if (!a->input.empty())
      add_(*a);
  } else if (auto *f = std::get_if<Fields>(&r)) {
    execute(*f);
  } else if (auto *o = std::get_if<Output>(&r)) {
    kernel_.outputs.push_back(std::move(*o));
  } else {
    ended_ = true;
    finish();
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
  for (View *v : fields.views())
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
  pipeline_ = &upload_();
}

void Driver::submit() {
  std::string text = module_->text();
  if (!text.empty())
    pipeline_->submit(std::move(text));
}

void Driver::finish() {
  if (!pipeline_)
    upload();
  pipeline_->end_trace(instructionCounter_);
  // A run killed in sync keeps this line.
  pipeline_->summary(std::cout);
  submit();
}

} // namespace nki
