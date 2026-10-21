# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API:
  - `runtime.h`
  - `tensor.h`
  - `pipeline.h`
- `src/`: the implementation; no PJRT type leaves it.
- `passes/`: the pass stage of the `custom-` compilers, and its `fsim-passes` tool.
- `third_party/`: what `setup.sh` sets up:
  - `pjrt/<entry>/`: per PJRT entry, the PJRT C API header and the plugin
  - `mlir/`: the MLIR and StableHLO headers in `include/` and static libraries in `lib/`, at StableHLO's pins
  - `dist/`: the CPU plugin and toolchain tarballs, and the GPU wheels

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
| `taidl`   | runs one module at the end of the run and throws on a second   |
| `trigger` | flushes at the soft trigger                                    |
| `overlap` | flushes at the soft trigger and compiles on background threads |

Outside `overlap`, `submit` runs each module before it returns.
`Knobs::parse` consumes the run-time knobs from a command line:

| knob          | default | what it sets              |
| ------------- | ------- | ------------------------- |
| `--trigger N` | 10000   | the soft trigger op count |
| `--overlap T` | 4       | the compile thread count  |

A knob the build lacks, or 0 threads, prints the refusal and exits 1.
`Pipeline::soft_trigger(ops)` tells a driver whether a graph of `ops` ops, the prologue counted, flushes now.
`Pipeline::stats()` gives each module's compile and execute ms.
In `overlap` its `waits` sums the executor's idle ms, each wait counted once it takes a module:

- `tracing_ms`: idle while the next module is not yet submitted, for diagnosis only
- `compile_ms`: idle while the next module is submitted but not yet compiled, the run's compile overhead

The pipeline clocks the run from its construction, or from the last `Pipeline::begin_run()`:

- `Pipeline::end_trace(n)` ends the driver's trace of `n` instructions.
- `Pipeline::summary(out)` prints the run's summary so far as one line, and flushes it.
- A driver prints the summary at its trace's end, so a run killed in `sync` keeps it, and again after its last `sync`.

```text
fsim stats: tracing=<s> overhead=<s> simulation=<s> tracing_overhead=<s> compile_overhead=<s> wall=<s> trace=<n> modules=<n>
```

Times are seconds to 3 decimals, and a value not yet known prints `-`:

| value              | what it is                                                      | known after               |
| ------------------ | --------------------------------------------------------------- | ------------------------- |
| `tracing`          | the clock at `end_trace`                                        | `end_trace`               |
| `overhead`         | `wall` minus `simulation`                                       | a clean `sync` of modules |
| `simulation`       | the modules' execute time                                       | a clean `sync` of modules |
| `tracing_overhead` | `overhead` minus `compile_overhead`                             | a clean `sync` of modules |
| `compile_overhead` | `waits.compile_ms` in `overlap`, else the modules' compile time | a clean `sync` of modules |
| `wall`             | the clock at `summary`                                          | any `sync`, failed or not |
| `trace`            | `end_trace`'s instruction count                                 | `end_trace`               |
| `modules`          | the submitted module count                                      | a clean `sync` of modules |

- A clean `sync` of modules returned, with at least one module submitted and none since.
- Outside `overlap`, `submit` compiles and runs its module, so a submit inside the trace counts in `tracing`.

The pass stage is `fsim_passes`, a shared object with only `passes.h`'s C entry visible, so its static MLIR never meets a driver's LLVM.
It takes a module's text and returns a versioned StableHLO artifact, which PJRT compiles in place of the text:

- parse the text into the `func` and `stablehlo` dialects
- CSE, the stock StableHLO target-independent optimization, canonicalize, CSE
- the library's own passes, `fsim::addPasses` in `passes/fsim_passes.cc`, in order, with canonicalize and CSE after the first, and `fsim-byte-grids`, canonicalize and CSE again at the end:
  - `fsim-byte-grids` sinks a bitcast between a `[..., K]` byte grid and `K`-byte elements below the slices, reshapes and updates that use it.
    Byte tensors stay bytes, and bitcasts act on tiles.
    An update of a 2-D byte tensor's `[P, M, K]` grid runs in the byte tensor's own shape.
  - `fsim-forward-slices` walks a slice up its chain of updates, past each one it misses.
    An update with constant indices that holds its window hands the slice its tile.
    A slice that misses every update reads the chain's head.
  - `fsim-dead-updates` walks a returned value up its chain of single-use updates.
    An update the later ones overwrite is dropped.
- `tf.aliasing_output` on every `@main` argument, so each state tensor updates in place
- serialize at the StableHLO version jax 0.11.2's XLA accepts

A failed phase fails the module's compile with the phase's diagnostics.
`fsim-passes` runs the stage over a module file, prints the op counts and each phase's ms, and writes the artifact, or the text with `--text`:

- `--trace` prints each pass's ms and the op count after it.
- `--histogram` prints the final op count per op name.

```bash
./docker/exec.sh "fsim/build/fsim-passes <in.mlir> -o <out.bc> [--text] [--trace] [--histogram]"
```

`setup.sh` takes two flags, and without either sets up the default entry and the toolchain:

| flag          | what it sets up                                                   |
| ------------- | ----------------------------------------------------------------- |
| `--pjrt NAME` | one of the four entries above, default `jax-0.11.2-cpu`           |
| `--mlir`      | the MLIR toolchain at StableHLO `v1.20.0` and that tag's LLVM pin |

- A `-cpu` entry unpacks its plugin from its tarball, and builds it for 1-2 h when no tarball is found.
- A `-cuda12` entry fetches its plugin and CUDA as PyPI wheels pinned by sha256 into `dist/`, then unpacks them.
- The toolchain unpacks from its tarball, and builds from source for about 10 min when no tarball is found.
- A part already unpacked from the same tarball is skipped.

A `-cpu` plugin or the toolchain looks for its tarball in this order:

1. `dist/`
2. `$FSIM_DIST_DIR`, copied into `dist/`
3. `$FSIM_DIST_URL/<tarball>`, fetched with curl into `dist/`
4. a build from source, packed into `dist/`

| variable        | default          | what it sets                                      |
| --------------- | ---------------- | ------------------------------------------------- |
| `FSIM_DIST_DIR` | `/opt/fsim/dist` | a directory of prebuilt tarballs, as an image has |
| `FSIM_DIST_URL` | unset            | a base url of prebuilt tarballs, as a release has |

- Steps 2 and 3 take only a tarball whose sha256 `setup.sh` pins, and skip an unpinned one.
- A tarball there whose sha256 differs from its pin fails the setup, never falling through to the next step.
- A failed download from `$FSIM_DIST_URL` fails the setup too, never falling through to the source build.
- `setup.sh` pins the `jax-0.11.2-cpu` and `jax-0.4.33-cpu` plugins and the toolchain on x86_64.

```bash
./scripts/setup.sh          # set up the default entry and the toolchain
./scripts/build.sh fsim     # build into fsim/build/
```
