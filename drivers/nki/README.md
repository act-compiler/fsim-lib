# The NKI driver

`drivers/nki/` is fsim-lib's driver for NKI, the kernel language and ISA of AWS Trainium.
A recorder runs a kernel under NKI's own frontend and writes one text record per instruction into a trace.
The C++ `fsim-nki` reads the trace and simulates them over the library in `fsim/`.
The design is based on the TAIDL specification of NKI 0.6.0 ([ACT Ecosystem](https://github.com/act-compiler/act): [TAIDL](https://github.com/act-compiler/taidl)) and the [AWS NKI documentation, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/).
Paths here are relative to the repo root.

## Running it

```bash
./scripts/build.sh nki [--compiler C] [--variant V]    # drivers/nki/build/<compiler>-<variant>/fsim-nki
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --inputs DIR --outputs DIR [--trigger N] [--overlap T] [<trace>|-]"
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --validate [<trace>|-]"
```

- `build.sh nki` builds `fsim-nki` against the library in `fsim/`, at one compiler and variant, `custom-cpu-overlap` by default.
- `fsim-nki` reads a trace file, or stdin for `-` or no trace, loads each input from `--inputs` and writes each output to `--outputs`.
- It ends with its evaluation log, fsim's summary line, which [design.md](design.md) explains:

  ```
  fsim stats: tracing=0.000 overhead=0.066 simulation=0.007 tracing_overhead=0.003 compile_overhead=0.063 wall=0.073 trace=11 modules=1
  ```

- `--trigger` and `--overlap` are the library's knobs.
- `--validate` parses the trace and runs its memory-model checks, which [design.md](design.md) lists, without simulating it.
- It takes no `--inputs` or `--outputs`, and ends with `checked records=N allocs=M bank_unchecked=K`.

## Design

[design.md](design.md) covers the record format, the C++ layers and the semantics.
