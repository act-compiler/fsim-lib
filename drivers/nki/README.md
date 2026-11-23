# The NKI driver

`drivers/nki/` is fsim-lib's driver for NKI, the kernel language and ISA of AWS Trainium.
A recorder runs a kernel under NKI's own frontend and writes one text record per instruction into a trace.
The C++ `fsim-nki` reads the trace and simulates them over the library in `fsim/`.
The design is based on the TAIDL specification of NKI 0.6.0 ([ACT Ecosystem](https://github.com/act-compiler/act): [TAIDL](https://github.com/act-compiler/taidl)) and the [AWS NKI documentation, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/).
Paths here are relative to the repo root.

## Running it

```bash
./scripts/build.sh nki [--compiler C] [--variant V]    # drivers/nki/build/<compiler>-<variant>/fsim-nki
./scripts/bench.sh gen --isa nki [--bench PATH]...     # each case's kernel.trace, inputs and golden
./scripts/bench.sh run --isa nki [--compiler C] [--variant V] [--bench PATH]... [--trigger N] [--overlap T] [--num-samples N] [--sampling S]
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --inputs DIR --outputs DIR [--trigger N] [--overlap T] [--num-samples N] [--sampling S] [<trace>|-]"
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --validate [<trace>|-]"
```

- `build.sh nki` builds `fsim-nki` against the library in `fsim/`, at one compiler and variant, `custom-cpu-overlap` by default.
- `bench.sh gen --isa nki` records each case once into `kernel.trace` in its folder under `benchmarks/nki/`.
- `bench.sh run --isa nki` hands each case's `kernel.trace` to that build's `fsim-nki`, and checks each sample's output against its input set's golden.
  A sample is one simulation of the trace on one input set, one value of every input tensor with its golden.
- A case's outputs go to `out/` in its folder under `benchmarks/nki/` until checked, leaving only `out/draws.bin`, and its log to `<compiler>-<variant>[-<trigger>[-<overlap>]][-n<num-samples>][-modulo].log` in the same folder.
- `fsim-nki` reads a trace file, or stdin for `-` or no trace.
- Each `--inputs` file `<name>.bin` holds its input's tensor from each of the case's input sets, back to back.
- `fsim-nki` simulates `--num-samples` samples, each with its input set drawn by `--sampling`:
  - `random`, the default: a uniform draw with replacement, from a fixed seed
  - `modulo`: sample `k` takes input set `k mod M` of the case's `M` input sets, so `--num-samples 1` runs input set 0
- The count rounds up to whole batches, a batch being the samples run together at the library's batch size `B`, which `fsim/README.md` defines.
- Each `--outputs` file `<name>.bin` holds every sample's output back to back, and `draws.bin` each sample's input-set index.
- `fsim-nki` ends with its evaluation log, fsim's summary line, which [design.md](design.md) explains:

  ```
  fsim stats: tracing=0.000 overhead=0.066 simulation=0.007 tracing_overhead=0.003 compile_overhead=0.063 wall=0.073 trace=11 modules=1 samples=1 batch_size=1
  ```

- `--trigger`, `--overlap` and `--num-samples` are the library's knobs, and `--sampling` the driver's.
- `--validate` parses the trace and runs its memory-model checks, which [design.md](design.md) lists, without simulating it.
- It takes no `--inputs`, `--outputs` or `--sampling`, and ends with `checked records=<n> allocs=<n> bank_unchecked=<n>`.

## Design

[design.md](design.md) covers the record format, the C++ layers and the semantics.
