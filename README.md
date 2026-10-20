# fsim-lib

fsim-lib is a scalable functional simulator for AI accelerators.
It is a common simulation library that ISA drivers sit over, each handing the library its ISA's tensor instructions.
The library builds StableHLO lazily and compiles and runs it through XLA's PJRT C API.

Under review at a major systems conference.

## Layout

- `fsim/`: the library; `fsim/README.md` documents it.
- `drivers/nki/`: the NKI driver; `drivers/nki/README.md` documents it.
- `docker/`: the image and its wrappers; every command runs through them.
- `benchmarks/nki/`: the nkilib benchmark cases and their generator under `gen/`.
- `scripts/`: the host entry points over `docker/exec.sh`:
  - `setup.sh`: the setup of the library's pinned PJRT entry and MLIR toolchain into `fsim/third_party/`
  - `build.sh`: the build of a target, `fsim` for the library or `nki` for the NKI driver at a `--compiler` and `--variant`
  - `clean.sh`: the removal of generated files, only files git ignores
  - `bench.sh`: the generation of the benchmark cases
- `format/`: the formatter and its per-language configs.

## Running it

Everything runs in the container; nothing is installed on the host.

```bash
./docker/build.sh                 # build the image, fsim-lib:latest-<arch>
./scripts/setup.sh                # set up the default PJRT entry and the MLIR toolchain
./scripts/build.sh fsim           # build the library into fsim/build/
```

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
./scripts/setup.sh --pjrt jax-0.11.2-cuda12   # set up one GPU entry
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

`./scripts/bench.sh gen` generates the nki benchmark cases in the image's nki env.
Each case is one nkilib kernel at one shape, one core's tensor-parallel shard of a Llama 3 or Llama 4 model.

- `benchmarks/nki/gen/nkilib/cases.py` lists the cases, and each case's folder under `benchmarks/nki/` holds a `.gitkeep`.
- `benchmarks/nki/suite.py` checks the list against the folders before any case is generated.
- gen compiles each case's kernel for trn2 through nki's compiler into the folder's `kernel.neff`.
- It writes the seeded inputs to `inputs/<tensor>.bin` and nkilib's torch reference output to `golden/<tensor>.bin`.
- git ignores the generated files.
- `--bench PATH` picks cases by name, directory or name prefix, and repeats; without one, gen regenerates every case.
- The first failing case stops the run.

```bash
./scripts/bench.sh gen                                # regenerate every case
./scripts/bench.sh gen --bench nkilib/attention_tkg   # regenerate one kernel's cases
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
