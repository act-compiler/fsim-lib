# The NKI driver

This design document is AI-generated, based on the TAIDL specification of NKI 0.6.0 ([ACT Ecosystem](https://github.com/act-compiler/act): [TAIDL](https://github.com/act-compiler/taidl)) and the [AWS NKI documentation, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/).

`drivers/nki/` is fsim-lib's driver for NKI, the kernel language and ISA of AWS Trainium, built here as `fsim-nki`.
Its binary is a trace, one text record per instruction, written by a backend module that runs the kernel under nki's own frontend.
`fsim-nki` decodes the records, keeps the ISA's non-tensor state, and hands every tensor instruction to the library as StableHLO.
It follows TAIDL's model of an ISA: an `Accelerator` declares data models and instructions, and each instruction has semantics over them ([accelerator.py](https://github.com/act-compiler/taidl/blob/v2.14/taidl/accelerator.py)).
Like act-oracle's test oracle, a functional simulator generated from a TAIDL spec, it lowers each instruction to StableHLO and runs it through XLA ([act-oracle](https://github.com/act-compiler/act-oracle/blob/v2.10/README.md)).
Paths here are relative to the repo root.

## Running it

```bash
./scripts/build.sh nki [--compiler C] [--variant V]    # drivers/nki/build/<compiler>-<variant>/fsim-nki
./scripts/bench.sh gen --isa nki [--bench PATH]...     # each case's kernel.trace, inputs and golden
./scripts/bench.sh run --isa nki [--compiler C] [--variant V] [--bench PATH]... [--trigger N] [--overlap T]
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --inputs DIR --outputs DIR [--trigger N] [--overlap T] [<trace>|-]"
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --validate [<trace>|-]"
```

The benchmarks run in one mode: gen records each case's trace once, and run reads the file.

- `bench.sh gen --isa nki` records each case through `drivers/nki/record` into `kernel.trace`, beside `kernel.neff`, `inputs/` and `golden/`.
- `bench.sh run --isa nki` hands each case's `kernel.trace` to `fsim-nki` as a file.
- `build.sh nki` builds `fsim-nki` and the library from `fsim/` at the compiler and variant `--compiler` and `--variant` pick, which `fsim/README.md` lists.
- The default is `custom-cpu` at `overlap`, built into `drivers/nki/build/custom-cpu-overlap/`.
- Each compiler and variant builds into its own directory, so these variant builds sit beside the default:
  - `--variant eager`
  - `--variant taidl`
  - `--variant trigger`
- `bench.sh run --isa nki` runs one build, picked by the same `--compiler` and `--variant`; `README.md`'s Bench lists its flags.
- It resolves cases as `gen` does and checks `out/out.bin` against `golden/out.bin`.
- The check is the max relative error, at most 3e-2 for bf16 and 1e-3 for f32.
- Each element's relative error divides by at least a floor of max |golden| times 0.1 for bf16, or 0.01 for f32.
- A run logs to `<case>/<compiler>-<variant>[-<trigger>[-<overlap>]].log`, with a knob per one the variant takes.
- `fsim-nki` reads a trace file, or stdin for `-` or no trace; the runner always passes the file.
- It loads `<inputs>/<name>.bin` for each input allocation and writes `<outputs>/<name>.bin` for each output.
- `--trigger` and `--overlap` are the library's knobs.
- An undecoded instruction kind, a bad record or a refused view fails the run loudly.

A run's evaluation log is fsim's summary line, here an `rmsnorm_tkg` case's under `overlap`:

```
fsim stats: tracing=0.000 overhead=0.066 simulation=0.007 tracing_overhead=0.003 compile_overhead=0.063 wall=0.073 trace=11 modules=1
```

- [fsim/README.md](../../fsim/README.md) defines its fields.
- The run's clock starts after the state's upload, at the first instruction, so it leaves out the header, the allocations and the input loads.
- The trace ends at `end`, with `trace` the instructions run.
- The line prints at `end`, so a run killed in the final sync keeps it, and again after the outputs are written.
- `bench.sh run` reads the last line into its status line.

`--validate` checks a trace without simulating it:

- It parses each record and runs the memory-model checks below, with instructions left undecoded, no inputs and nothing lowered.
- It checks that each `output` names a declared allocation.
- It ends with one line, `checked records=N allocs=M bank_unchecked=K`.
- `bank_unchecked` counts the `nc_matmul` dsts whose `+s(...)` offset leaves the bank check to run time.

## The record format

A trace is ASCII text, one record per line, fields separated by single spaces, the first field the record kind.
The parser is a tokenizer plus one decode function per instruction kind, which takes its fields by name.
An unknown kind, an unknown field or a missing one is an error.

```
trace    := header record* "end"
header   := "nki-trace" version "target=trn2" "lnc=1" "hbm=" bytes
record   := alloc | instr | control | output
alloc    := "alloc" id arena "@" addr dtype "[" dims "]" placed ("name=" name)? ("input=" name)?
instr    := kind (field "=" value)*
control  := "reg.load" reg view | "reg.set" reg int | "loop" reg int int int | "endloop"
output   := "output" name id
value    := view | number | "true" | "false" | "none" | word | "[" (number ("," number)*)? "]" | affine
view     := arena "@" addr ":" dtype pattern dyn?
pattern  := "[" "[" stride "," count "]" ("," "[" stride "," count "]")* "]"
dyn      := "+s(" view ")*" int | "+r" int "*" int
affine   := "aff(" pattern "," int ")"
addr     := bytes | partition "," bytes
id       := "%" int
reg      := "r" int
```

### Header

`nki-trace 1 target=trn2 lnc=1 hbm=<bytes>` is the first line.

- `1` is the format version; the parser refuses any other.
- `hbm=` is the HBM arena's capacity, which the recorder sets before the kernel runs.
- The capacity is the kernel's highest HBM byte plus the inputs, the outputs and a scratch reserve.
- The reserve is the inputs' and outputs' size, or the `reserve` bytes `record(build, out, reserve)` takes.
- The recorder rounds it up to a multiple of 4 bytes, and `fsim-nki` refuses any other capacity.
- The header precedes the stream but needs the output sizes, so the recorder makes a dry pass first, about 2x the record time.

### Allocations

`alloc %3 sbuf@0,64 bf16[128,1,32] kernel` declares tensor `%3` in SBUF at partition 0, byte 64 per partition, 128 partitions of 32 bf16.

- `arena` is `hbm`, `sbuf` or `psum`; `private_hbm` and `shared_hbm` both map to `hbm`.
- `addr` is a byte offset for `hbm`, and `partition,byte` for `sbuf` and `psum`.
- `placed` is `kernel` when the kernel allocated the tensor at an address, and `driver` when the recorder placed it.
- `input=<name>` marks a kernel input; `fsim-nki` loads `<inputs>/<name>.bin` there before the first instruction.
- `name=` keeps the tensor's name when it has one.

A kernel address is used as given, so aliasing, capacity and bank checks are exact for those tensors.
nkilib's allocator passes `(partition, byte)` to `nl.ndarray` unless it runs in its auto-allocation mode, which passes no address ([nki-library at 92d11f6, allocator.py lines 412-421](https://github.com/aws-neuron/nki-library/blob/92d11f6/src/nkilib_src/nkilib/core/utils/allocator.py#L412-L421)).
The cte kernels, mlp_tkg and moe place almost every tensor explicitly; the norms, rope, cumsum, the other tkg kernels and attention_tkg, through `create_auto_alloc_manager`, place none.
The recorder places a tensor without an address:

- It takes the lowest free gap in the arena that fits, first fit.
- HBM and SBUF align the tensor to its dtype's size, and PSUM to a 2 KiB bank.
- So a PSUM tensor that fits in one bank never straddles a bank boundary.
- The inputs come first in HBM.
- A tensor is released when the kernel drops its handle, and a stack tensor when its scope closes.
- A placement past an arena's capacity fails the record, so a driver-placed tensor is bounded by its placement.
- Capacity checks for such tensors reflect this placement, not the compiler's.
- Driver-placed tensors, such as the identity matrix, go above the kernel's highest explicit address, so they never overlap kernel-placed tensors.
- A multi-buffer of `num_slots>1` at an explicit address is refused.

### Views

A view is one operand: `sbuf@0,64:bf16[[1,128],[0,1],[1,32]]`.

- `arena@addr` is the first byte, as in the allocation; `dtype` is the operand's dtype, a bitcast of the arena's bytes.
- The pattern is `[[stride, count], ...]`, outer dimension first.
- For `sbuf` and `psum` the first entry is the partition dimension, its stride in partitions, 0 for a broadcast.
- Every other entry's stride is in elements of `dtype` within the partition, or within HBM, 0 for a broadcast.
- The operand's shape is the counts.
- The recorder rewrites a view's element offset and pattern over its allocation's row into this form.
- It refuses a view whose partition stride is not a whole number of partitions.
- `+s(<view>)*<k>` adds the index the inner view reads, times `k` elements, at run time.
- `+r<N>*<k>` is reserved for the same with register `r<N>`, and `fsim-nki` refuses it until M3.
- An index view is a single `i32`; a multi-element `+s(...)` view is refused until M3.
- A `+s(...)` offset moves within a partition row, never across partitions.
- The recorder refuses a `+s(...)` offset on an SBUF or PSUM partition dimension until M3.
- `dtype` is one of `bf16`, `f32`, `i32`, `u8`, `u16`, `u32`.

### Instructions

An instruction record is its kind followed by the builtin's keyword fields in their order, as `field=value`.
A kind is the name of the nki backend builtin the recorder implements, as `dir(nki._backends.simulator)` lists them.
The recorder drops `name`, `priority` and `dge_mode`, which carry no semantics, and keeps every other field.

| value      | spelling                                                       |
| ---------- | -------------------------------------------------------------- |
| view       | as above                                                       |
| number     | a Python int or float repr, e.g. `0.000244140625`              |
| bool, None | `true`, `false`, `none`                                        |
| enum       | its name: `op=multiply`, `engine=vector`, `reduce_cmd=idle`    |
| list       | `axes=[1,0]`, `shuffle_mask=[0,1,...]`                         |
| affine map | `pred=aff([[1,128],[-1,512]],511)`, the pattern and its offset |

### Control records

Loops stay unrolled: every Python loop runs at record time and leaves its iterations in the stream.
Four records are reserved in the format, so a rolled loop or a register loop is an extension, not a format change.
`fsim-nki` refuses each of them until M3:

| record                            | meaning                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| `reg.load r<N> <view>`            | submit, sync, read the view's first int32 into the register          |
| `reg.set r<N> <int>`              | set the register                                                     |
| `loop r<N> <start> <stop> <step>` | push a loop; the register counts from `start`                        |
| `endloop`                         | step the register and jump to the loop's first record while `< stop` |

### End of kernel

`output out %2` names a returned tensor; `fsim-nki` writes its bytes to `<outputs>/out.bin` after the final sync.
`end` is the last line; a stream without it is an error.

### Example

The records of `rmsnorm_tkg/llama3_8b_tp4`, abridged, as the recorder emits them:

```
nki-trace 1 target=trn2 lnc=1 hbm=49152
alloc %0 hbm@0 bf16[1,1,4096] driver input=input
alloc %1 hbm@8192 bf16[1,4096] driver input=gamma
alloc %2 hbm@16384 bf16[128,1,32] driver name=out
alloc %3 sbuf@0,0 bf16[128,1,32] driver
dma_copy dst=sbuf@0,0:bf16[[1,128],[32,1],[32,1],[1,32]] src=hbm@0:bf16[[32,128],[4096,1],[4096,1],[1,32]] oob_is_err=true engine=dma
alloc %5 sbuf@0,128 f32[128,1] driver
memset dst=sbuf@0,128:f32[[1,128],[1,1]] value=1e-06 engine=unknown
alloc %7 sbuf@0,644 f32[128,1,32] driver
activation dst=sbuf@0,644:f32[[1,128],[32,1],[1,32]] op=square data=sbuf@0,0:bf16[[1,128],[32,1],[1,32]] bias=0.0 scale=1.0 reduce_op=none reduce_res=none reduce_cmd=idle
tensor_reduce_arith dst=sbuf@0,772:f32[[1,128],[1,1]] op=add data=sbuf@0,644:f32[[1,128],[32,1],[1,32]] num_r_dim=1 negate=false keepdims=false
nc_matmul dst=psum@0,0:f32[[1,128],[1,1]] stationary=sbuf@0,132:f32[[1,128],[1,128]] moving=sbuf@0,772:f32[[1,128],[1,1]] is_transpose=false row_pos=none col_pos=none perf_mode=none accumulate=none
output out %2
end
```

## Layout

The driver is a pipeline: `Parser` parses each line into a record, and `Driver` runs it over an `Accelerator` as it arrives.
Decode is one free function per instruction kind, which the decode table in `src/isa/Instruction.cpp` dispatches by the kind's name.
Each decode builds an `Instruction` whose `semantics(Accelerator&, Module&)` emits its StableHLO into a `Module`.
`Driver` owns fsim's `Runtime` and `Pipeline`, and runs the modules through them.
Parsing, accelerator state, code generation and instruction semantics each stay in one layer, and each layer includes only those before it:

1. `src/trace/`
2. `src/accelerator/` and `src/stablehlo/`, each over `trace/` alone
3. `src/isa/`
4. `src/Driver.{h,cpp}` and `src/main.cpp`

| path                                  | role                                                                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `record/backend.py`                   | the backend `nki._backends._activate_backend` installs: tensor builtins to records, allocations to placement            |
| `record/place.py`                     | the per-arena first-fit placement for allocations without an address                                                    |
| `record/record.py`                    | `record(build, out)`: runs the kernel under the backend and writes the stream to a file or a pipe                       |
| `src/trace/View.h`                    | `View`, one operand, and `Dyn`, its `+s(...)` offset; `Arena`, `Dtype` and a pattern's `Entry`                          |
| `src/trace/Fields.h`                  | `Fields`: an instruction record's values, which its decode takes by name and `done()` checks are all taken              |
| `src/trace/Kernel.{h,cpp}`            | `Kernel`: the header and the outputs; `Alloc` and `Output`                                                              |
| `src/trace/Parser.{h,cpp}`            | one line to one record: tokenizer, view and pattern parser; it refuses the reserved records                             |
| `src/isa/Instruction.h`               | `Instruction` and its `semantics(Accelerator&, Module&)`; `decoder(kind)` and the `decode<Kind>(Fields&)`               |
| `src/isa/Instruction.cpp`             | the decode table from a kind to its decode; a later milestone's kind is refused by name                                 |
| `src/isa/<Instruction>.cpp`           | one kind's `Instruction` subclass with `const` fields, and its decode, named after the public nki.isa instruction       |
| `src/accelerator/Accelerator.{h,cpp}` | `Accelerator`: the arenas' host copies, the allocation table, the reduce registers, the PSUM written mask               |
| `src/accelerator/Limits.{h,cpp}`      | the memory-model checks: an allocation within its arena, `nc_matmul`'s arenas and its dst in one PSUM bank              |
| `src/stablehlo/Module.{h,cpp}`        | `Module`: state parameters, view `read` and `write`, ops, and `text()`, which ends the module and returns its text      |
| `src/Driver.{h,cpp}`                  | `Driver`: `run()` over a file or stdin; `resolve`; upload, soft trigger, submit, sync, read-back and the evaluation log |
| `src/main.cpp`                        | the `fsim-nki` command line                                                                                             |
| `CMakeLists.txt`                      | `add_subdirectory(../../fsim)`, links `fsim`                                                                            |

## From a record to StableHLO

The library runs a fixed set of state tensors through modules.
They are the driver's data models, in TAIDL's sense of a declared buffer an instruction addresses ([declarations](https://github.com/act-compiler/taidl/blob/v2.14/docs/declarations.md)).
`hbm` plays TAIDL's `d0`, the off-chip memory as a flat buffer of bytes.
The driver's state is four:

| state         | type               | what it is                                                                                                                                                          |
| ------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hbm`         | `ui8[1, hbm]`      | the HBM arena at the header's capacity                                                                                                                              |
| `sbuf`        | `ui8[128, 229376]` | SBUF, 224 KiB per partition                                                                                                                                         |
| `psum`        | `ui8[128, 16384]`  | PSUM, 8 banks of 2 KiB per partition                                                                                                                                |
| `reduce_regs` | `f32[128, 1]`      | the activation engine's reduce registers ([nki.isa.activation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.activation.html)) |

- SBUF is modelled at the documented 224 KiB per partition.
- nki's compile path rejects some SBUF ranges below 224 KiB; the docs give no usable size.

One instruction is one `semantics` call, which reads its views, computes, and writes its destination view.
A view is an NKI access pattern over a region of one data model.
It lowers to the ops the [StableHLO spec](https://openxla.org/stablehlo/spec) defines.

- `read`: the arena is reshaped to a `[P, R/K, K]` byte grid and bitcast to the view's `K`-byte dtype, a `[P, R/K]` element grid.
- Unit counts drop out of the pattern, and zero strides fold to one element.
- A single-level pattern is one strided [slice](https://openxla.org/stablehlo/spec#slice) of the grid, its partitions and elements together.
- A nested pattern, each stride a multiple of the next and the entries disjoint, is a slice of its window first.
- The window is then delinearized into a reshape, one dimension per entry, as MLIR's [affine.delinearize_index](https://mlir.llvm.org/docs/Dialects/Affine/#affinedelinearize_index-affineaffinedelinearizeindexop) splits a flat index, and sliced to the counts.
- Any other pattern falls back to a gather with a constant index.
- A zero stride becomes a broadcast back to the full shape.
- The result is `[count_0, ..., count_n]` in the view's dtype.
- compute: the ops the numpy simulator spells, over f32 converts of the operands.
- `write`: a nested pattern at partition stride 1 is a dynamic_update_slice of the arena's window, after an update of the delinearized window when the pattern leaves gaps.
- Any other pattern falls back to a scatter; either way the grid is bitcast back to bytes.
- A write through a broadcast view is refused.
- The library's byte-grid pass keeps bytes as bytes below the bitcasts.
- An f32 to integer convert is round-to-nearest-even, saturating, NaN to the maximum, as nki.simulate casts.

Control and state:

- `Driver::run` asks `Pipeline::soft_trigger` for the module's op count before each instruction.
- When it fires, `Module::text` ends the module and the driver submits its text, and the module starts afresh.
- Under `overlap`, submitted modules compile on the library's threads while the driver keeps decoding.
- `Driver::resolve` resolves each view's `+s(...)` offset before the instruction's decode and semantics.
- It submits the module, syncs, and reads the index from the arena's host copy, so the offset enters the module as a constant address.
- `Accelerator::resolve` then checks every view against its arena and its dtype's alignment.
- `nc_matmul` resolves `accumulate` at trace time from a per-byte PSUM written mask in `Accelerator`:
  - A PSUM allocation, kernel-placed or driver-placed, clears the mask over its bytes.
  - Any write to PSUM, by a matmul or not, sets its bytes.
  - `true` adds onto the tile.
  - `false` clears the mask over the newest PSUM allocation holding the destination, then overwrites.
  - `none` adds where the tile was written and overwrites elsewhere.
  - `none` with `is_transpose=true` always overwrites.
  - This is nki.simulate's per-tensor rule while no two live PSUM tensors overlap, since a view names no tensor.
- The reduce register lanes index from the operand's partition start, and `reduce_res` reads them.
- `reduce_cmd=reset` zeroes them, `reset_reduce` stores the reduction, and `reduce` adds it on.
- `activation` refuses a `reduce_op` other than `add` with `reduce` or `reset_reduce`, since the ISA reduces by add only ([nki.isa.activation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.activation.html)).
- The suite reaches `reduce_cmd=reset_reduce` with a `reduce_res` in `activation` and `tensor_scalar_reduce`, through attention_cte and qkv_cte.
- Each instruction runs as its record arrives, and `instructionCounter_` counts them for the summary's `trace`.
- After `end`, the driver ends the trace and prints the summary, then submits, syncs, writes the outputs and prints it again.

## Memory and dtypes

- Memory is three byte arenas at trn2's sizes; every tensor is a region of one, at the address its allocation record names.
- Two allocations at the same address alias.
- A view that crosses its arena's partition row or its end is refused.
- A view whose byte offset is not a multiple of its dtype's size is refused.
- The arenas start zeroed, where nki's simulator fills fresh storage with NaN.
- A dtype is a view: the same bytes read as bf16 or as u16 are the same bytes, and nothing is stored in a wider type; f32 to bf16 rounds to nearest even on `convert`.
- Integer arithmetic on the GpSimd engine stays in its own dtype, as nki.simulate computes it, and everything else computes in f32.

## Memory-model checks

`fsim-nki` checks each allocation record as it arrives and each `nc_matmul` record before it runs, in `src/accelerator/Limits.cpp`.
A violation fails the run as `line N: <what>`, naming the record's line, the tensor and the limit.
The limits are trn2's, from the NKI 2.32 docs:

- An allocation spans at most partitions 0 to 127 of SBUF or PSUM ([memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/get-started/about/memory-hierarchy-overview.html#neuroncore-internal-memory)).
- An SBUF allocation spans at most bytes 0 to 229375 of a partition, the documented 224 KiB ([Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium2_arch.html#neuroncore-v3-compute-engine-updates)).
- A PSUM allocation spans at most bytes 0 to 16383 of a partition, 16 KiB ([Trainium/Inferentia2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html#near-memory-accumulation-in-psum)).
- An `nc_matmul` dst lies within one 2 KiB PSUM bank per partition ([Trainium/Inferentia2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html#near-memory-accumulation-in-psum)).
- An `nc_matmul` reads `stationary` and `moving` from SBUF and writes `dst` to PSUM ([nki.isa.nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_matmul.html)).

- The arena bounds hold for every allocation; a kernel-placed one is checked at the kernel's address.
- A driver-placed tensor already fails at record time when its placement overflows, and `fsim-nki` bounds it again.
- The bank check holds for every dst, since the recorder places no PSUM tensor that fits a bank across a bank boundary.
- A dst with a `+s(...)` offset is bank-checked once resolved, and `--validate` counts it in `bank_unchecked`.
- The `xfail` bench source holds one kernel per limit.
  Its gen logs nki's compile path and `nki.simulate` on each kernel, and its run expects `fsim-nki` to reject each:

  ```bash
  ./scripts/bench.sh gen --isa nki --bench xfail
  ./scripts/bench.sh run --isa nki --bench xfail
  ```

## Semantics

Semantics follow nki's numpy simulator, re-expressed as StableHLO, except where the chip is observed to disagree, and then the chip wins.

- Where the NKI 2.32 docs are silent, call a pattern undefined, or contradict each other, the driver follows nki.simulate.
- So it accepts the documented contradictions the suite relies on, as nki.simulate and the compiler do: `nc_matmul` at `col_pos` (start_column) 32, 64 or 96 ([nki.isa.nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_matmul.html)), and a transposing `nc_matmul` whose dst is in its input's dtype, bf16 ([nki.isa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_transpose.html)).
- `nc_matmul` with `accumulate=none` adds onto a PSUM tile that a `memset` or copy wrote, as nki.simulate does; the docs call this undefined on trn2.
- Transcendentals are XLA's f32 ops, and the hardware engines may round differently, as the NKI simulator guide's [Numerical Precision](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/nki_simulator.html#numerical-precision) section says of nki.simulate.
- `reduce_cmd=reduce` and `reset_reduce` on `activation` reduce by add, the one `reduce_op` the ISA documents.
- `reduce_cmd=load_reduce` on `activation` follows nki's simulator, which leaves the reduce registers unchanged; it is to be checked on the chip at M2.

## Coverage

The suite's thirteen kernels reach seventeen instruction kinds at trn2, each with one keyword set, no register load and no register loop.

| kind                   | milestone | kernels                                                 |
| ---------------------- | --------- | ------------------------------------------------------- |
| `dma_copy`             | M1        | all                                                     |
| `memset`               | M1        | all but rope and output_proj                            |
| `activation`           | M1        | norms, qkv, mlp, attention, moe                         |
| `tensor_tensor_arith`  | M1        | norms, rope, qkv_tkg, mlp, attention, moe               |
| `tensor_reduce_arith`  | M1        | norms, qkv_tkg, attention, moe all_expert               |
| `nc_matmul`            | M1        | all but rope and cumsum                                 |
| `tensor_copy`          | M2        | rope, cumsum, qkv_cte, output_proj, mlp, attention, moe |
| `tensor_tensor_scan`   | M2        | cumsum                                                  |
| `tensor_scalar_arith`  | M2        | qkv_cte, attention, moe                                 |
| `affine_select`        | M2        | qkv_cte, mlp, attention, moe                            |
| `dma_transpose`        | M2        | attention_cte                                           |
| `reciprocal`           | M2        | attention                                               |
| `tensor_scalar_reduce` | M2        | attention_cte                                           |
| `select_reduce`        | M2        | attention_tkg                                           |
| `nc_stream_shuffle`    | M2        | attention_tkg, moe                                      |
| `stream_transpose`     | M2        | attention_tkg, moe selective                            |
| `local_gather`         | M3        | moe selective                                           |

Within an M1 kind, M1 decodes these values, and any other fails at decode until its milestone:

| field                                      | M1 decodes                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| `activation` `op`                          | copy, exp, log, sqrt, rsqrt, tanh, sigmoid, relu, square, abs, reciprocal    |
| `activation` `reduce_op`                   | add; with `reset` or `load_reduce`, also multiply, maximum, minimum          |
| `tensor_tensor_arith` `op`                 | the arithmetic ops: add, subtract, multiply, divide, maximum, minimum, power |
| `tensor_tensor_arith` `op`, gpsimd integer | add, subtract, multiply, maximum, minimum                                    |
| `tensor_reduce_arith` `op`                 | add, multiply, maximum, minimum                                              |
| `nc_matmul` `perf_mode`                    | none                                                                         |

- M1: rmsnorm_tkg and layernorm_tkg, 11 and 19 instructions over six kinds.
- M2: the eleven straight-line kernels.
- M3: moe_tkg selective, with the materialized `+s(...)` views, and all_expert.

## Scope and limits

- One target, trn2 at lnc=1: one core, no `program_id` sharding, no `sendrecv` or `core_barrier`.
- The dtypes above, with bf16, f32, i32 and u8 as the kernels' data; no fp8, no MX and no collectives, on which the backend raises.
- Execution is one program order, as in nki's simulator: no engines, semaphores or scheduling, so no race is modelled.
- Inputs are `shared_hbm`, as a compiled kernel sees them; this is what lets all_expert record.
- A kernel that converts a register to a Python value cannot be recorded; the backend raises, as nki's compiler frontend does.
- Batching over input sets is a later `fsim-nki` flag.

## References

The driver's terms follow these, pinned to the versions it was written against:

- TAIDL v2.14, [accelerator.py](https://github.com/act-compiler/taidl/blob/v2.14/taidl/accelerator.py): the `Accelerator`, its instructions and their semantics.
- TAIDL v2.14, [declarations](https://github.com/act-compiler/taidl/blob/v2.14/docs/declarations.md): data models and buffers, `d0` as off-chip byte memory, and regions.
- TAIDL v2.14, [instructions](https://github.com/act-compiler/taidl/blob/v2.14/docs/instructions.md): how an instruction is declared.
- act-oracle v2.10, [README](https://github.com/act-compiler/act-oracle/blob/v2.10/README.md): the test oracle, a functional simulator, with its kernels and loops.
- act-oracle v2.10, [operations](https://github.com/act-compiler/act-oracle/blob/v2.10/docs/operations.md): how operations lower to StableHLO.
- [StableHLO specification](https://openxla.org/stablehlo/spec): the ops the driver emits.
- [MLIR affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/#affinedelinearize_index-affineaffinedelinearizeindexop): `affine.delinearize_index`, the split a nested view's reshape follows.
- [NKI CPU Simulator guide, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/nki_simulator.html#numerical-precision): nki.simulate's numerical precision.
- [nki.isa.activation, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.activation.html): `reduce_regs`, `reduce_cmd` and `reduce_op`.
