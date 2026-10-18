// The pipeline's threads: a pool of compile workers fills modules in any
// order, and one executor runs them strictly in submission order. All shared
// state sits behind one mutex and one condition variable. Outside the overlap
// variant there are no threads: submit() compiles and executes in place.
#include "fsim/pipeline.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#ifdef FSIM_VARIANT_OVERLAP
#include <condition_variable>
#include <deque>
#include <thread>
#endif

#include "compile_options.h"
#include "pjrt.h"

namespace fsim {
namespace {

using Clock = std::chrono::steady_clock;

// The knobs Knobs::parse kept.
Knobs knobs;

// Prints a refusal of the command line and exits 1.
[[noreturn]] void refuse(std::string_view message) {
  std::cerr << "fsim: " << message << "\n";
  std::exit(1);
}

// Parses one knob's count; refuses anything but a whole number.
template <typename T> T count(std::string_view knob, const char *text) {
  T value{};
  std::string_view digits = text ? text : "";
  auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
  if (digits.empty() || error != std::errc() || end != digits.data() + digits.size())
    refuse(std::string(knob) + " takes a count");
  return value;
}

#ifdef FSIM_VARIANT_OVERLAP
// One submitted module, from text to executable.
struct Module {
  std::string text;
  PJRT_LoadedExecutable *executable = nullptr;
  std::exception_ptr error;
  bool compiled = false;
};
#endif

// A signature as PJRT type and dims, e.g. "type 11 [4,8]".
std::string signature(PJRT_Buffer_Type type, const int64_t *dims, size_t num_dims) {
  std::string text = "type " + std::to_string(type) + " [";
  for (size_t d = 0; d < num_dims; ++d)
    text += (d ? "," : "") + std::to_string(dims[d]);
  return text + "]";
}

#ifdef FSIM_COMPILER_CUSTOM
// The library's StableHLO passes; none yet.
void passes(std::string &) {
}
#endif

} // namespace

size_t Stats::total_bytes() const {
  size_t bytes = 0;
  for (const ModuleStats &module : modules)
    bytes += module.bytes;
  return bytes;
}

struct Pipeline::Impl {
  const Runtime::Impl &rt;
  std::vector<Tensor> state;
  std::vector<PJRT_Buffer *> buffers;
  const std::string options = compile_options();
  const Clock::time_point zero = Clock::now();
  const uint32_t trigger;

  // Guarded by mutex; changed signals every update.
  mutable std::mutex mutex;
  Stats stats;
  std::exception_ptr error;
#ifdef FSIM_VARIANT_OVERLAP
  std::condition_variable changed;
  std::deque<Module> modules;
  size_t next_compile = 0;
  size_t next_execute = 0;
  bool stop = false;

  std::vector<std::thread> workers;
  std::thread executor;
#endif

  Impl(const Runtime::Impl &rt, std::vector<Tensor> state);
  ~Impl();

  double now() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - zero).count();
  }
#ifdef FSIM_VARIANT_OVERLAP
  void compile_loop();
  void execute_loop();
#else
  void run(std::string text, ModuleStats &stats);
#endif
  PJRT_LoadedExecutable *compile(std::string text) const;
  void check_outputs(PJRT_LoadedExecutable *loaded) const;
  void execute(PJRT_LoadedExecutable *executable);
};

Pipeline::Impl::Impl(const Runtime::Impl &rt, std::vector<Tensor> state)
    : rt(rt), state(std::move(state)), trigger(knobs.trigger.value_or(10000)) {
  try {
    // One device buffer per state tensor.
    buffers.reserve(this->state.size());
    for (const Tensor &tensor : this->state) {
      PJRT_Client_BufferFromHostBuffer_Args args{};
      args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      args.client = rt.client;
      args.data = tensor.data;
      args.type = buffer_type(tensor.type);
      args.dims = tensor.dims.data();
      args.num_dims = tensor.dims.size();
      args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
      args.device = rt.device;
      rt.check(rt.api->PJRT_Client_BufferFromHostBuffer(&args), "PJRT_Client_BufferFromHostBuffer");
      buffers.push_back(args.buffer);
      rt.await(args.done_with_host_buffer);
    }
#ifdef FSIM_VARIANT_OVERLAP
    for (unsigned i = 0, threads = knobs.threads.value_or(4); i < threads; ++i)
      workers.emplace_back(&Impl::compile_loop, this);
    executor = std::thread(&Impl::execute_loop, this);
#endif
  } catch (...) {
    // Undo a partial construction: stop, join, free.
#ifdef FSIM_VARIANT_OVERLAP
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    changed.notify_all();
    for (std::thread &worker : workers)
      worker.join();
#endif
    for (PJRT_Buffer *buffer : buffers)
      rt.destroy(buffer);
    throw;
  }
}

Pipeline::Impl::~Impl() {
#ifdef FSIM_VARIANT_OVERLAP
  {
    std::lock_guard lock(mutex);
    stop = true;
  }
  changed.notify_all();
  for (std::thread &worker : workers)
    worker.join();
  executor.join();
  for (Module &module : modules)
    if (module.executable)
      rt.destroy(module.executable);
#endif
  for (PJRT_Buffer *buffer : buffers)
    rt.destroy(buffer);
}

#ifdef FSIM_VARIANT_OVERLAP

void Pipeline::Impl::compile_loop() {
  std::unique_lock lock(mutex);
  for (;;) {
    changed.wait(lock, [&] { return stop || next_compile < modules.size(); });
    if (stop)
      return;
    size_t index = next_compile++;
    std::string text = std::move(modules[index].text);
    bool skip = error != nullptr;
    lock.unlock();

    // Compile outside the lock; a dead pipeline skips the work.
    double begin = now();
    PJRT_LoadedExecutable *executable = nullptr;
    std::exception_ptr failure;
    if (!skip) {
      try {
        executable = compile(std::move(text));
      } catch (...) {
        failure = std::current_exception();
      }
    }
    double end = now();

    lock.lock();
    modules[index].executable = executable;
    modules[index].error = failure;
    modules[index].compiled = true;
    stats.modules[index].compile_begin_ms = begin;
    stats.modules[index].compile_end_ms = end;
    changed.notify_all();
  }
}

void Pipeline::Impl::execute_loop() {
  std::unique_lock lock(mutex);
  for (;;) {
    changed.wait(lock, [&] {
      return stop || (!error && next_execute < modules.size() && modules[next_execute].compiled);
    });
    if (stop)
      return;
    Module &module = modules[next_execute];
    if (module.error) {
      error = module.error;
      changed.notify_all();
      continue;
    }
    PJRT_LoadedExecutable *executable = module.executable;
    module.executable = nullptr;
    lock.unlock();

    // Execute outside the lock; only this thread touches the buffers.
    double begin = now();
    std::exception_ptr failure;
    try {
      execute(executable);
    } catch (...) {
      failure = std::current_exception();
    }
    rt.destroy(executable);
    double end = now();

    lock.lock();
    stats.modules[next_execute].execute_begin_ms = begin;
    stats.modules[next_execute].execute_end_ms = end;
    ++next_execute;
    if (failure)
      error = failure;
    changed.notify_all();
  }
}
#else
// Compiles and executes one module in place.
void Pipeline::Impl::run(std::string text, ModuleStats &stats) {
  stats.compile_begin_ms = now();
  PJRT_LoadedExecutable *executable = compile(std::move(text));
  stats.compile_end_ms = now();
  stats.execute_begin_ms = stats.compile_end_ms;
  try {
    execute(executable);
  } catch (...) {
    rt.destroy(executable);
    throw;
  }
  rt.destroy(executable);
  stats.execute_end_ms = now();
}
#endif

PJRT_LoadedExecutable *Pipeline::Impl::compile(std::string text) const {
#ifdef FSIM_COMPILER_CUSTOM
  passes(text);
#endif
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = const_cast<char *>(text.data());
  program.code_size = text.size();
  program.format = "mlir";
  program.format_size = 4;
  PJRT_Client_Compile_Args args{};
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  args.client = rt.client;
  args.program = &program;
  args.compile_options = options.data();
  args.compile_options_size = options.size();
  rt.check(rt.api->PJRT_Client_Compile(&args), "PJRT_Client_Compile");
  try {
    check_outputs(args.executable);
  } catch (...) {
    rt.destroy(args.executable);
    throw;
  }
  return args.executable;
}

// The module must return exactly the state: count, types and dims.
void Pipeline::Impl::check_outputs(PJRT_LoadedExecutable *loaded) const {
  PJRT_LoadedExecutable_GetExecutable_Args get{};
  get.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get.loaded_executable = loaded;
  rt.check(rt.api->PJRT_LoadedExecutable_GetExecutable(&get),
           "PJRT_LoadedExecutable_GetExecutable");
  try {
    PJRT_Executable_OutputElementTypes_Args types{};
    types.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
    types.executable = get.executable;
    rt.check(rt.api->PJRT_Executable_OutputElementTypes(&types),
             "PJRT_Executable_OutputElementTypes");
    if (types.num_output_types != state.size())
      throw std::runtime_error("fsim: module returns " + std::to_string(types.num_output_types) +
                               " results, the pipeline holds " + std::to_string(state.size()) +
                               " state tensors");
    PJRT_Executable_OutputDimensions_Args dims{};
    dims.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
    dims.executable = get.executable;
    dims.num_outputs = state.size();
    rt.check(rt.api->PJRT_Executable_OutputDimensions(&dims), "PJRT_Executable_OutputDimensions");
    const int64_t *result_dims = dims.dims;
    for (size_t i = 0; i < state.size(); ++i) {
      const Tensor &tensor = state[i];
      PJRT_Buffer_Type type = buffer_type(tensor.type);
      bool same = types.output_types[i] == type && dims.dim_sizes[i] == tensor.dims.size() &&
                  std::equal(tensor.dims.begin(), tensor.dims.end(), result_dims);
      if (!same)
        throw std::runtime_error("fsim: module result " + std::to_string(i) + " is " +
                                 signature(types.output_types[i], result_dims, dims.dim_sizes[i]) +
                                 ", state tensor " + std::to_string(i) + " is " +
                                 signature(type, tensor.dims.data(), tensor.dims.size()));
      result_dims += dims.dim_sizes[i];
    }
  } catch (...) {
    rt.destroy(get.executable);
    throw;
  }
  rt.destroy(get.executable);
}

void Pipeline::Impl::execute(PJRT_LoadedExecutable *executable) {
  PJRT_ExecuteOptions options{};
  options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  std::vector<PJRT_Buffer *> results(buffers.size());
  PJRT_Buffer *const *argument_lists[1] = {buffers.data()};
  PJRT_Buffer **output_lists[1] = {results.data()};
  PJRT_Event *done[1] = {nullptr};
  PJRT_LoadedExecutable_Execute_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.executable = executable;
  args.options = &options;
  args.argument_lists = argument_lists;
  args.num_devices = 1;
  args.num_args = buffers.size();
  args.output_lists = output_lists;
  args.device_complete_events = done;
  rt.check(rt.api->PJRT_LoadedExecutable_Execute(&args), "PJRT_LoadedExecutable_Execute");
  try {
    rt.await(done[0]);
  } catch (...) {
    for (PJRT_Buffer *result : results)
      rt.destroy(result);
    throw;
  }

  // The results become the state; the old buffers go.
  for (PJRT_Buffer *buffer : buffers)
    rt.destroy(buffer);
  buffers = std::move(results);
}

int Knobs::parse(int argc, char **argv) {
  // Compacts argv over every other argument.
  int kept = 1;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--trigger")
      knobs.trigger = count<uint32_t>(arg, argv[++i]);
    else if (arg == "--overlap")
      knobs.threads = count<unsigned>(arg, argv[++i]);
    else
      argv[kept++] = argv[i];
  }
  argv[kept] = nullptr;

  // Refuses a knob this build lacks.
#if defined(FSIM_VARIANT_EAGER) || defined(FSIM_VARIANT_TAIDL)
  if (knobs.trigger)
    refuse("this build has no soft trigger");
#endif
#ifndef FSIM_VARIANT_OVERLAP
  if (knobs.threads)
    refuse("this build compiles on no threads");
#endif
  if (knobs.threads == 0u)
    refuse("--overlap needs at least one compile thread");
  return kept;
}

Pipeline::Pipeline(Runtime &runtime, std::vector<Tensor> state)
    : impl_(std::make_unique<Impl>(*runtime.impl_, std::move(state))) {
}

Pipeline::~Pipeline() = default;

bool Pipeline::soft_trigger([[maybe_unused]] size_t ops) const {
#if defined(FSIM_VARIANT_EAGER)
  // Any op past the state's prologue flushes.
  return ops > impl_->state.size();
#elif defined(FSIM_VARIANT_TAIDL)
  return false;
#else
  return ops >= impl_->trigger;
#endif
}

void Pipeline::submit(std::string module) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->error)
    std::rethrow_exception(impl_->error);
#ifdef FSIM_VARIANT_TAIDL
  // One module per run; a second is the pipeline's error.
  if (!impl_->stats.modules.empty()) {
    impl_->error = std::make_exception_ptr(std::runtime_error(
        "fsim: a second module; this build runs one module at the end of the run"));
    std::rethrow_exception(impl_->error);
  }
#endif
  ModuleStats stats;
  stats.bytes = module.size();
  impl_->stats.modules.push_back(stats);
#ifdef FSIM_VARIANT_OVERLAP
  impl_->modules.emplace_back().text = std::move(module);
  impl_->changed.notify_all();
#else
  // A failure stays the pipeline's error.
  try {
    impl_->run(std::move(module), impl_->stats.modules.back());
  } catch (...) {
    impl_->error = std::current_exception();
    throw;
  }
#endif
}

void Pipeline::sync() {
  {
    std::unique_lock lock(impl_->mutex);
#ifdef FSIM_VARIANT_OVERLAP
    impl_->changed.wait(
        lock, [&] { return impl_->error || impl_->next_execute == impl_->modules.size(); });
#endif
    if (impl_->error)
      std::rethrow_exception(impl_->error);
  }

  // Nothing executes: copy every state buffer to its host memory.
  const Runtime::Impl &rt = impl_->rt;
  for (size_t i = 0; i < impl_->buffers.size(); ++i) {
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = impl_->buffers[i];
    args.dst = impl_->state[i].data;
    args.dst_size = byte_size(impl_->state[i]);
    rt.check(rt.api->PJRT_Buffer_ToHostBuffer(&args), "PJRT_Buffer_ToHostBuffer");
    rt.await(args.event);
  }
}

Stats Pipeline::stats() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}

} // namespace fsim
