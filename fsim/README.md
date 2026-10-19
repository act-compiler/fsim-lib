# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API:
  - `runtime.h`
  - `tensor.h`
  - `pipeline.h`
- `src/`: the implementation; no PJRT type leaves it.
- `third_party/`: per PJRT entry, `pjrt/<entry>/` holds the PJRT C API header and the plugin `setup.sh` sets up, and `dist/` the CPU plugin tarballs and GPU wheels.

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

- A `custom-` compiler runs the library's StableHLO passes before compile, none yet.
- The configure fails when the entry's plugin is not set up.
- A CPU build runs on one CPU device, and a GPU build on the first device the plugin reports.

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

`setup.sh` takes `--pjrt NAME`, one of the four entries above, default `jax-0.11.2-cpu`.

- A `-cpu` entry unpacks its plugin from its `dist/` tarball, and builds it for 1-2 h when the tarball is absent.
- A `-cuda12` entry fetches its plugin and CUDA as PyPI wheels pinned by sha256 into `dist/`, then unpacks them.

```bash
./scripts/setup.sh          # set up the --pjrt entry
./scripts/build.sh fsim     # build into fsim/build/
```
