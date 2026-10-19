# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API:
  - `runtime.h`
  - `tensor.h`
  - `pipeline.h`
- `src/`: the implementation; no PJRT type leaves it.
- `third_party/`: per PJRT entry, `pjrt/<entry>/` holds the PJRT C API header and the CPU plugin `setup.sh` sets up, and `dist/` the plugin tarballs.

A consumer adds it with `add_subdirectory(fsim)` and links `fsim`.
Three cache variables fix the build:

| variable        | default      | what it sets                                                                             |
| --------------- | ------------ | ---------------------------------------------------------------------------------------- |
| `FSIM_PJRT`     | `jax-0.11.2` | the entry whose header the library builds against and whose plugin path it bakes in      |
| `FSIM_VARIANT`  | `overlap`    | when modules flush and where they run                                                    |
| `FSIM_COMPILER` | `custom`     | `xla` or `custom`; `custom` runs the library's StableHLO passes before compile, none yet |

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

`setup.sh` takes `--pjrt NAME`, `jax-0.11.2` or `jax-0.4.33`, default `jax-0.11.2`.
Without the entry's tarball, it builds its plugin for 1-2 h.

```bash
./scripts/setup.sh          # set up the --pjrt entry
./scripts/build.sh fsim     # build into fsim/build/
```
