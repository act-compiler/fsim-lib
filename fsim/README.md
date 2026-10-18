# fsim

The common simulation library: a static C++ library that compiles StableHLO modules with a PJRT plugin and runs them over a fixed set of state tensors.

- `include/fsim/`: the public API, `runtime.h`, `tensor.h` and `pipeline.h`.
- `src/`: the implementation; no PJRT type leaves it.
- `third_party/`: the PJRT C API header and CPU plugin, fetched by `setup.sh`.

A consumer adds it with `add_subdirectory(fsim)` and links `fsim`.
The plugin path is the cache variable `FSIM_PJRT_PLUGIN`, baked in at build time.

```bash
./scripts/setup.sh          # fetch the pinned plugin and header
./scripts/build.sh fsim     # build into fsim/build/
```
