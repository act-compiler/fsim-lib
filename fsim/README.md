# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API:
  - `runtime.h`
  - `tensor.h`
  - `pipeline.h`
- `src/`: the implementation; no PJRT type leaves it.
- `passes/`: the pass stage of the `custom-` compilers, and its `fsim-passes` tool.
- `third_party/`: what `setup.sh` sets up:
  - `mlir/`: the MLIR and StableHLO headers in `include/` and static libraries in `lib/`, at StableHLO's pins
  - `pjrt/<entry>/`: per PJRT entry, the PJRT C API header and the plugin
  - `dist/`: the toolchain and CPU plugin tarballs, and the GPU wheels

A consumer adds it with `add_subdirectory(fsim)` and links `fsim`.
Two cache variables fix the build:

| variable        | default      | what it sets                                |
| --------------- | ------------ | ------------------------------------------- |
| `FSIM_VARIANT`  | `overlap`    | when modules flush and where they run       |
| `FSIM_COMPILER` | `custom-cpu` | the compiler, its device and its PJRT entry |

The compiler picks the entry whose header the library builds against and whose plugin path it bakes in:

| compiler       | device | PJRT entry          |
| -------------- | ------ | ------------------- |
| `xla-0.11-cpu` | CPU    | `jax-0.11.2-cpu`    |
| `xla-0.4-cpu`  | CPU    | `jax-0.4.33-cpu`    |
| `custom-cpu`   | CPU    | `jax-0.11.2-cpu`    |
| `xla-0.11-gpu` | GPU    | `jax-0.11.2-cuda12` |
| `xla-0.4-gpu`  | GPU    | `jax-0.4.33-cuda12` |
| `custom-gpu`   | GPU    | `jax-0.11.2-cuda12` |

- A `custom-` compiler runs the pass stage on every module before compile; its time counts in the module's compile.
- The configure of a `custom-` compiler fails when the MLIR toolchain is not set up.
- The configure fails when the entry's plugin is not set up.
- A CPU build runs on one CPU device, and a GPU build on the first device the plugin reports.
- A `jax-0.11.2-cuda12` build first loads the CUDA libraries jax preloads, from its entry's `nvidia/`.
- A `custom-gpu` build creates its client with `memory_fraction` 0.95, its allocator limited to 95% of total GPU memory, the device limit `Pipeline::batch_size` uses.
  The `xla-` GPU builds keep the plugin's default.

Each compiler compiles with these XLA option overrides:

| compiler       | XLA options                                                                           |
| -------------- | ------------------------------------------------------------------------------------- |
| `xla-0.11-cpu` | `xla_backend_optimization_level=1`, `xla_cpu_copy_insertion_use_region_analysis=True` |
| `xla-0.4-cpu`  | `xla_backend_optimization_level=1`                                                    |
| `custom-cpu`   | `xla_backend_optimization_level=1`, `xla_cpu_copy_insertion_use_region_analysis=True` |
| `xla-0.11-gpu` | `xla_gpu_autotune_level=0`, `xla_gpu_enable_triton_gemm=False`                        |
| `xla-0.4-gpu`  | `xla_gpu_autotune_level=0`, `xla_gpu_enable_triton_gemm=False`                        |
| `custom-gpu`   | `xla_gpu_autotune_level=0`, `xla_gpu_enable_triton_gemm=False`                        |

A plugin of an older minor version than the header loads when it has every PJRT function the library calls.
The variants flush and run as follows:

| variant   | flush and run                                                  |
| --------- | -------------------------------------------------------------- |
| `eager`   | flushes at every op past the state's prologue                  |
| `taidl`   | runs one module at each trace's end and throws on a second     |
| `trigger` | flushes at the soft trigger                                    |
| `overlap` | flushes at the soft trigger and compiles on background threads |

Outside `overlap`, `submit` runs each module before it returns.
`Knobs::parse` consumes the run-time knobs from a command line:

| knob              | default | what it sets                                                                      |
| ----------------- | ------- | --------------------------------------------------------------------------------- |
| `--trigger N`     | 10000   | the soft trigger op count                                                         |
| `--overlap T`     | 4       | the compile thread count                                                          |
| `--num-samples N` | 1       | the samples per run, each one simulation of the driver's trace over its own state |

Each of these prints the refusal and exits 1:

- a knob the build lacks
- 0 threads
- 0 samples
- a `--num-samples` above 1 outside the `custom-` compilers, whose pass stage runs the vmap pass

A run of `N` samples runs in batches, each `B` samples simulated together, where `B` is the batch size `Pipeline::batch_size(runtime, state)` gives for one sample's state tensors:

- On the CPU, `B` is the smaller of `N` and 16; on the GPU, `B` is `N`.
- A batch of `B` fits when `3 x B x` the state's bytes are within the available host memory and `2 x B x` them within the device's `bytes_limit`.
- On the CPU the host holds the batch's state three times: in the driver's host memory, the state buffers and an execute's results.
- The device holds the last two; the host's three are checked on the CPU and the GPU alike.
- On the CPU the device's limit is the available host memory; on the GPU it is the device's reported limit, and a device that reports none throws.
- Past either limit, `B` falls to the largest power of two that fits.
- `Pipeline::batch_size` throws when one sample does not fit.
- At `N` = 1, `B` is 1 and neither limit is checked.
- `Pipeline::num_samples()` gives `N`.

Each state tensor's host memory holds the batch's samples back to back, and above a batch size of 1 the device holds it as `[B, dims]`:

- `Pipeline(runtime, state, batch_size)` takes the batch size.
- The sample at batch index `m` starts from its part at byte `m x byte_size(tensor)`.
- `sync` hands each of the batch's samples back to its own part.
- Above a batch size of 1, a mid-trace `sync`, one before `end_trace`, throws, so a driver that needs one runs at a batch size of 1.

At an `N` above `B`, the pipeline keeps every executable it runs, in submission order, until it is destroyed, begins the next trace or runs a mid-trace `sync`.
`Pipeline::replay()` replays them over the state's host memory, whose every byte the driver first loads with the next batch's samples, since `sync` left the last outputs there:

- It uploads every state tensor anew from the constructor's host memory.
- It executes the kept executables in order on the caller's thread, adding each one's execute ms to its stats.
- A `sync` then hands the samples back, as after a `submit`.

`Pipeline::begin_trace()` begins the next trace on the same pipeline, for a driver that traces each sample again:

- It uploads every state tensor anew from the host memory.
- The kept executables go, and nothing replays after it.
- The trace's `tracing` clock starts there, and its modules add to the stats.

`Pipeline::exclude(ms)` leaves `ms` of a driver's loads and stores out of the run's clock.

`Pipeline::soft_trigger(ops)` tells a driver whether a graph of `ops` ops, the prologue counted, flushes now.
`Pipeline::stats()` gives each module's compile and execute ms.
In `overlap` its `waits` sums the executor's idle ms, each wait counted once it takes a module:

- `tracing_ms`: idle while the next module is not yet submitted, for diagnosis only
- `compile_ms`: idle while the next module is submitted but not yet compiled, the run's compile overhead

The pipeline clocks the run from its construction, or from the last `Pipeline::begin_run()`:

- `Pipeline::end_trace(n)` ends the driver's trace of `n` instructions.
- `Pipeline::summary(out)` prints the run's summary so far as one line, and flushes it.
- A driver prints the summary at its trace's end, so a run killed in `sync` keeps it, and again after its last `sync`.
  A driver that begins a next trace prints one line per trace, then a last one.

```text
fsim stats: tracing=<s> overhead=<s> simulation=<s> tracing_overhead=<s> compile_overhead=<s> wall=<s> trace=<n> modules=<n> samples=<n> batch_size=<n>
```

Times are seconds to 3 decimals, and a value not yet known prints `-`:

| value              | what it is                                                      | known after               |
| ------------------ | --------------------------------------------------------------- | ------------------------- |
| `tracing`          | the clock at `end_trace`, summed over the run's traces          | `end_trace`               |
| `overhead`         | `wall` minus `simulation`                                       | a clean `sync` of modules |
| `simulation`       | the modules' execute time, every replay's included              | a clean `sync` of modules |
| `tracing_overhead` | `overhead` minus `compile_overhead`                             | a clean `sync` of modules |
| `compile_overhead` | `waits.compile_ms` in `overlap`, else the modules' compile time | a clean `sync` of modules |
| `wall`             | the clock at `summary`, less every `exclude`                    | any `sync`, failed or not |
| `trace`            | `end_trace`'s instruction count                                 | `end_trace`               |
| `modules`          | the submitted module count                                      | a clean `sync` of modules |
| `samples`          | the samples simulated, `B` per batch, a driver's padding too    | construction, each batch  |
| `batch_size`       | the batch size the pipeline runs at                             | construction              |

- A clean `sync` of modules returned, with at least one module submitted and no submit, replay or `end_trace` since.
- Outside `overlap`, `submit` compiles and runs its module, so a submit inside the trace counts in `tracing`.

The pass stage is `fsim_passes`, a shared object with only `passes.h`'s C entry visible, so its static MLIR never meets a driver's LLVM.
It takes a module's text and returns a versioned StableHLO artifact, which PJRT compiles in place of the text:

- parse the text into the `func` and `stablehlo` dialects
- CSE, the stock StableHLO target-independent optimization, canonicalize, CSE
- the library's own passes, `fsim::addPasses` in `passes/fsim_passes.cc`, in order, with canonicalize and CSE after the first, and `fsim-byte-grids`, canonicalize and CSE again before `fsim-order-reads`:
  - `fsim-byte-grids` sinks a bitcast between a `[..., K]` byte grid and `K`-byte elements below the slices, reshapes and updates that use it.
    Byte tensors stay bytes, and bitcasts act on tiles.
    An update of a 2-D byte tensor's `[P, M, K]` grid runs in the byte tensor's own shape.
  - `fsim-forward-slices` walks a slice up its chain of updates, past each one it misses.
    An update with constant indices that holds its window hands the slice its tile.
    A slice that misses every update reads the chain's head.
  - `fsim-dead-updates` walks a returned value up its chain of single-use updates.
    An update the later ones overwrite is dropped.
  - `fsim-order-reads` threads a buffer through one `stablehlo.optimization_barrier` per read of it, beside that read, before the update that overwrites it.
    A read is threaded as the op its single uses lead to, which may be another update, so it stays fused there.
    When that op cannot move before the update, the read is threaded as itself.
    XLA then updates the buffer in place, with no copy of it.
    A read after its update moves before it, with the ops it needs.
    A buffer is left as is on a second update, a return, a read in another block, or a read whose move needs its update, a region or a side effect.
- `tf.aliasing_output` on every `@main` argument, so each state tensor updates in place
- above a batch size of 1, `fsim-vmap` gives `@main`'s arguments and results a leading batch dimension of size `B`, keeping the aliasing.
  A constant, or an elementwise op over unbatched operands, stays unbatched; an op whose rule batches its operands takes a broadcast of it.
  An op fails the module's compile unless it has a `kRules` entry in `passes/vmap.cc` or is a StableHLO op with MLIR's `Elementwise` trait.
- serialize at the StableHLO version jax 0.11.2's XLA accepts

A failed phase fails the module's compile with the phase's diagnostics.
`fsim-passes` runs the stage over a module file, prints the op counts and each phase's ms, and writes the artifact, or the text with `--text`:

- `--trace` prints each pass's ms and the op count after it.
- `--histogram` prints the final op count per op name.
- `--batch-size B` runs the vmap pass at batch size `B`.
- `passes/test/` holds one small module per group of vmap rules; CI runs each at `--batch-size 2`.

```bash
./docker/exec.sh "fsim/build/fsim-passes <in.mlir> -o <out.bc> [--text] [--trace] [--histogram] [--batch-size B]"
```

`setup.sh` takes names as arguments, each one part to set up:

| argument      | what it sets up                                                     |
| ------------- | ------------------------------------------------------------------- |
| `mlir`        | the MLIR toolchain at StableHLO `v1.20.0` and that tag's LLVM pin   |
| an entry name | one of the four entries above                                       |
| none          | the toolchain, every `-cpu` entry, and every `-cuda12` one on a GPU |

- Names repeat and combine, and only the named parts are set up.
- Without a name, a GPU is one `nvidia-smi` sees in the container.
- The toolchain or a `-cpu` plugin unpacks its tarball from `dist/`, else pinned from `/opt/fsim/dist`, else from a source build.
  That build takes about 10 min for the toolchain and 1-2 h for a plugin.
- A `-cuda12` entry fetches its plugin and CUDA as PyPI wheels pinned by sha256 into `dist/`, then unpacks them.
- A part already unpacked from the same tarball is skipped.
- An image tarball whose sha256 differs from its pin fails the setup.
- `setup.sh` runs on x86_64 only, and pins the toolchain and the `jax-0.11.2-cpu` and `jax-0.4.33-cpu` plugins.

```bash
./scripts/setup.sh          # set up the toolchain and every plugin this host runs
./scripts/build.sh fsim     # build into fsim/build/
```
