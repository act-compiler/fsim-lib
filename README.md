# fsim-lib

fsim-lib is a scalable functional simulator for AI accelerators.
It is a common simulation library that ISA drivers sit over, each handing the library its ISA's tensor instructions.
The library builds StableHLO lazily and compiles and runs it through XLA's PJRT C API.

Under review at a major systems conference.

## Layout

- `fsim/`: the library; `fsim/README.md` documents it.
- `drivers/nki/`: the NKI driver; `drivers/nki/README.md` documents it.
- `docker/`: the image and its wrappers; every command runs through them.
- `benchmarks/nki/`: the nkilib and xfail benchmark cases and their generators under `gen/`.
- `scripts/`: the host entry points over `docker/exec.sh`:
  - `setup.sh`: the setup of the library's pinned PJRT entry and MLIR toolchain into `fsim/third_party/`
  - `build.sh`: the build of a target, `fsim` for the library or `nki` for the NKI driver at a `--compiler` and `--variant`
  - `clean.sh`: the removal of generated files, only files git ignores
  - `bench.sh`: the generation of one ISA's benchmark cases and their runs on one build of its driver
- `format/`: the formatter and its per-language configs.
- `docs/`: reports; `docs/nki-simulate.md` compares `nki.simulate` with runs on a trn2 chip.
- `.github/workflows/`: the CI, which builds both targets, then generates and runs the nki suite.

## Running it

Everything runs in the container; nothing is installed on the host.

```bash
./docker/build.sh                 # build the image, fsim-lib:latest-<arch>, on the env image
./scripts/setup.sh                # set up the default PJRT entry and the MLIR toolchain
./scripts/build.sh fsim           # build the library into fsim/build/
```

- `docker/build.sh` pulls the env image `devanshdvj/fsim-lib-env` at `docker/fsim-env.env`'s `FSIM_ENV_TAG`, and builds it locally when the pull fails and `FSIM_ENV_TAG` equals `FSIM_ENV_BUILD_TAG`.
- The env image holds only open-source parts, and the build adds nki and its compiler from AWS's pip index.
- The env image holds no CUDA; a GPU host adds it through the GPU entry under Setup.
- `./docker/build.sh --rebuild` replaces an existing image.
- `docker/exec.sh` passes every GPU when the host has the nvidia runtime.

```bash
./docker/exec.sh "<cmd>"          # one command in a throwaway container
./docker/run.sh --persistent      # an interactive shell that survives
./docker/chown.sh                 # give a root-owned workspace back to the host user
```

## Setup

`./scripts/setup.sh` runs `fsim/setup.sh` in the container, and without a flag sets up both parts:

- `--pjrt NAME`: one PJRT entry, `jax-0.11.2-cpu` by default; `fsim/README.md` lists the entries.
- `--mlir`: the MLIR and StableHLO toolchain the `custom-` compilers need.

A part already in `fsim/third_party/` is skipped.
A `-cpu` plugin builds for 1-2 h when its tarball is absent from `fsim/third_party/dist/`, and the toolchain for about 10 min.

```bash
./scripts/setup.sh --pjrt jax-0.11.2-cuda12   # a GPU host adds CUDA: one GPU entry and its wheels
```

## Build

`./scripts/build.sh fsim` configures and builds the library into `fsim/build/`.
It builds at the default compiler and variant of the cache variables `fsim/README.md` lists.

`./scripts/build.sh nki` configures and builds the NKI driver into `drivers/nki/build/<compiler>-<variant>/fsim-nki`.
It builds the library from `fsim/` alongside, at the compiler and variant `--compiler` and `--variant` pick, `custom-cpu` and `overlap` by default.

```bash
./scripts/build.sh nki                       # build drivers/nki/build/custom-cpu-overlap/
./scripts/build.sh nki --variant trigger     # build drivers/nki/build/custom-cpu-trigger/
```

## Bench

`./scripts/bench.sh gen --isa nki` generates the nki benchmark cases in the image's nki env.
Each `nkilib` case is one nkilib kernel at one shape, one core's tensor-parallel shard of a Llama 3 or Llama 4 model.

- Each source's `benchmarks/nki/gen/<source>/cases.py` lists its cases, and each case's folder under `benchmarks/nki/` holds a `.gitkeep`.
- `benchmarks/nki/suite.py` checks the list against the folders before any case is generated.
- gen compiles each case's kernel for trn2 through nki's compiler into the folder's `kernel.neff`.
- It records the kernel through `drivers/nki/record` into `kernel.trace`, the file `fsim-nki` reads.
- It writes the seeded inputs to `inputs/<tensor>.bin` and nkilib's torch reference output to `golden/<tensor>.bin`.
- Each `xfail` case breaks one trn2 memory limit at an explicit address, and gen writes only its `kernel.trace` and inputs.
  Its gen line logs the compiler's rejection and whether `nki.simulate` runs it, and a compile that succeeds fails the case.
- git ignores the generated files.
- `--bench PATH`: the cases, by name, directory or name prefix; it repeats, and without one, gen regenerates every case.
- The first failing case stops the run.

```bash
./scripts/bench.sh gen --isa nki                                # regenerate every case
./scripts/bench.sh gen --isa nki --bench nkilib/attention_tkg   # regenerate one kernel's cases
```

`./scripts/bench.sh run --isa nki` runs the generated cases on one build of the NKI driver.
It hands each case's `kernel.trace` to that build's `fsim-nki`, and checks the output against the golden.

- `--compiler C`: the build's library compiler, `custom-cpu` by default; `fsim/README.md` lists the compilers.
- `--variant V`: the build's library variant, `overlap` by default.
- `--trigger N`: the soft trigger op count, 10000 by default, for the `trigger` and `overlap` variants only.
- `--overlap T`: the compile threads, 4 by default, for the `overlap` variant only.
- `--bench PATH`: the cases, picked as gen picks them; it repeats, and without one, run takes every case.
- `FSIM_WALL_TIMEOUT`: each case's wall seconds, 60 by default.
- The build is `drivers/nki/build/<compiler>-<variant>/fsim-nki`, which `./scripts/build.sh nki` builds.
- A case's output goes to `out/<tensor>.bin` in the case's folder, beside `golden/`.
- A case logs to `<compiler>-<variant>[-<trigger>[-<overlap>]].log` in its folder, with a knob per one the variant takes.
- The log's last line, also printed per case, is its status: `match`, `wrong`, `failing`, `N/A` or `not generated`.
- `N/A` marks a `taidl` build that stops at a second module.
- An `xfail` case is `xfail` when `fsim-nki` rejects it with the case's expected message, `xpass` when it accepts it, and `failing` otherwise.
- A simulation error marks its case `failing`, with the error in its log, and the run goes on.
- The run exits nonzero unless every case is `match` or `xfail`.

The status line also gives the case's worst relative error, `err`, and the fields of the last `fsim stats:` line in its log, times in seconds; [fsim/README.md](fsim/README.md) defines them:

| Field              | Meaning                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `tracing`          | the trace loop                                                        |
| `wall`             | the whole run                                                         |
| `simulation`       | every module's execute, summed                                        |
| `overhead`         | `wall` less `simulation`                                              |
| `compile_overhead` | the `overlap` executor's wait on compile, else every module's compile |
| `tracing_overhead` | `overhead` less `compile_overhead`, the output copy included          |
| `trace`            | the trace length, a count                                             |
| `modules`          | the modules run, a count                                              |

```bash
./scripts/bench.sh run --isa nki                              # run every case
./scripts/bench.sh run --isa nki --bench nkilib/rmsnorm_tkg   # run one kernel's cases
./scripts/bench.sh run --isa nki --bench xfail                # run the memory-limit cases
```

## Clean

`./scripts/clean.sh` removes only files git ignores, and never anything under `work/`.
Without a flag it removes every ignored `__pycache__/` directory.

- `--build`: also every ignored `build/` directory
- `--third-party`: also `fsim/third_party/`, its `dist/` tarballs included, and implies `--build`
- `--gen`: also every ignored file under `benchmarks/`, the generated cases, and implies `--build`
- `-n`: prints what would go and removes nothing

```bash
./scripts/clean.sh -n --build     # show what a build clean would remove
```

## Format

`format/format.sh` formats the paths it is given, or the whole tree without one.

- A whole-tree run refuses while unstaged changes are present, unless `FORMAT_SKIP_CLEAN_CHECK=1` is set.
- A formatter not on the `PATH` is skipped with a note; the image has them all.

```bash
./docker/exec.sh "./format/format.sh <paths>"   # format these paths
./docker/exec.sh "./format/format.sh"           # format the whole tree
```
