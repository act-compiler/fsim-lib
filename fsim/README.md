# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API, `runtime.h`, `tensor.h` and `pipeline.h`.
- `src/`: the implementation; no PJRT type leaves it.
- `third_party/`: per PJRT entry, `pjrt/<entry>/` holds the PJRT C API header and the CPU plugin `setup.sh` sets up, and `dist/` the plugin tarballs.

A consumer adds it with `add_subdirectory(fsim)` and links `fsim`.
The cache variable `FSIM_PJRT`, `jax-0.11.2` by default, names the entry whose header the library builds against and whose plugin path it bakes in.
A plugin of an older minor version than the header loads when it has every PJRT function the library calls.
The cache variable `FSIM_VARIANT`, `overlap` by default, fixes when modules flush and where they run: `eager` flushes at every op past the state's prologue, `taidl` runs one module at the end of the run and throws on a second, `trigger` flushes at the soft trigger, and `overlap` flushes at the soft trigger and compiles on background threads; outside `overlap`, `submit` runs each module before it returns.
The cache variable `FSIM_COMPILER`, `custom` by default, is `xla` or `custom`; `custom` runs the library's StableHLO passes before compile, none yet.
`Knobs::parse` consumes the run-time knobs from a command line: `--trigger N`, the soft trigger op count, 10000 by default, and `--overlap T`, the compile thread count, 4 by default; a knob the build lacks, or 0 threads, prints the refusal and exits 1.
`Pipeline::soft_trigger(ops)` tells a driver whether a graph of `ops` ops, the prologue counted, flushes now.

```bash
./scripts/setup.sh          # set up --pjrt NAME, jax-0.11.2 or jax-0.4.33, default jax-0.11.2; without its tarball, builds its plugin for 1-2 h
./scripts/build.sh fsim     # build into fsim/build/
```
