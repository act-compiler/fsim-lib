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
./scripts/bench.sh run --isa nki [--compiler C] [--variant V] [--bench PATH]... [--trigger N] [--overlap T] [--num-samples N] [--sampling S]
./docker/exec.sh "drivers/nki/build/custom-cpu-overlap/fsim-nki --inputs DIR --outputs DIR [--trigger N] [--overlap T] [--num-samples N] [--sampling S] [<trace>|-]"
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
- It resolves cases as `gen` does and checks each sample's part of `out/<name>.bin`.
  A sample is one simulation of the trace on one input set, one value of every input tensor with its golden.
- The part is checked against the part of `golden/<name>.bin` for the input set `out/draws.bin` gives that sample.
- The status line's `err` is the worst over every sample.
- The check is the max relative error, at most the dtype's tolerance below.
- Each element's relative error divides by at least a floor of max |golden| times the dtype's fraction:

  | Dtype | Max relative error | Floor fraction |
  | ----- | ------------------ | -------------- |
  | bf16  | 5e-2               | 0.1            |
  | f16   | 1e-2               | 0.1            |
  | f32   | 1e-3               | 0.01           |

- A run logs to `<case>/<compiler>-<variant>[-<trigger>[-<overlap>]][-n<num-samples>][-modulo].log`, with a knob per one the variant takes.
- `fsim-nki` reads a trace file, or stdin for `-` or no trace, holding stdin's text in memory; the runner always passes the file.
- Its batch loop, in `src/samples/`, reads the trace in full each time it traces.
- `<inputs>/<name>.bin` holds its input's tensor from each of the case's `M` input sets, back to back.
- `--sampling S` draws each simulated sample's input set:
  - `random`, the default: a uniform draw with replacement over the `M` input sets, from a fixed-seed `std::mt19937_64`, the same on every run with one C++ standard library
  - `modulo`: sample `k` takes input set `k mod M`, so `--num-samples 1` runs input set 0
- Loading a sample zeroes its state, then reads each input's tensor from its input set.
- Samples run together in batches of the library's batch size `B`, which `fsim/README.md` defines.
- The first batch traces; its first instruction builds the library's pipeline over the batch's state and starts the run's clock.
- Each later batch loads its samples and replays the first's executables.
- Extra samples pad the last batch to `B`, and every one is simulated and checked.
- Above one sample, a parse-only reading first looks for a `+s(...)` run-time offset.
  Such a trace runs at a batch size of 1: each sample traces alone, and the pipeline begins its next trace for it.
- `<outputs>/<name>.bin` holds the output of every simulated sample in order, and `<outputs>/draws.bin` each sample's input-set index as a little-endian uint32, under either `--sampling` value.
- `fsim-nki` refuses before the first output byte when the outputs pass half the free space of `--outputs`, or 1 GiB.
- `--trigger`, `--overlap` and `--num-samples` are the library's knobs, and `--sampling` the driver's.
- An undecoded instruction kind, a bad record or a refused view fails the run loudly.

A run's evaluation log is fsim's summary line, here an `rmsnorm_tkg` case's under `overlap`:

```
fsim stats: tracing=0.000 overhead=0.066 simulation=0.007 tracing_overhead=0.003 compile_overhead=0.063 wall=0.073 trace=11 modules=1 samples=1 batch_size=1
```

- [fsim/README.md](../../fsim/README.md) defines its fields.
- The run's clock starts after the state's upload, at the first instruction, and leaves out the header, the allocations and every input load and output store.
- The trace ends at `end`, with `trace` the instructions run.
- The line prints at each trace's `end`, so a run killed in a sync keeps it, and again after every sample's outputs are written.
- `bench.sh run` reads the last line into its status line.

`--validate` checks a trace without simulating it:

- It parses each record and runs the memory-model checks below, with instructions left undecoded, no inputs and nothing lowered.
- It takes no `--inputs`, `--outputs` or `--sampling`.
- It checks that each `output` names a declared allocation.
- It ends with one line, `checked records=<n> allocs=<n> bank_unchecked=<n>`.
- `bank_unchecked` counts the `nc_matmul` dsts whose `+s(...)` offset hides the bank, which a run refuses.

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
view     := arena id "@" addr ":" dtype pattern dyn?
pattern  := "[" "[" stride "," count "]" ("," "[" stride "," count "]")* "]"
dyn      := "+s(" view ")*" int | "+r" int "*" int
affine   := "aff(" pattern "," int ")"
addr     := bytes | partition "," bytes
id       := "%" int
reg      := "r" int
```

### Header

`nki-trace 2 target=trn2 lnc=1 hbm=<bytes>` is the first line.

- `2` is the format version; the parser refuses any other, version 1 included, and names the version it read.
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
- `input=<name>` marks a kernel input; `fsim-nki` loads there, before the first instruction, each sample's tensor of its input set from `<inputs>/<name>.bin`.
- `name=` keeps the tensor's name when it has one.

A kernel address is used as given, so aliasing, capacity and bank checks are exact for those tensors.
nkilib's allocator passes `(partition, byte)` to `nl.ndarray` unless it runs in its auto-allocation mode, which passes no address ([nki-library at 92d11f6, allocator.py lines 412-421](https://github.com/aws-neuron/nki-library/blob/92d11f6/src/nkilib_src/nkilib/core/utils/allocator.py#L412-L421)).
These kernels place almost every tensor explicitly:

- attention_cte
- mlp_cte
- qkv_cte
- mlp_tkg
- moe selective

These place none:

- output_proj_cte
- moe all_expert
- the matmul cases
- the norms
- rope
- cumsum
- qkv_tkg
- output_proj_tkg
- attention_tkg

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

A view is one operand: `sbuf%3@0,64:bf16[[1,128],[0,1],[1,32]]`.

- `%3` is the allocation the view is over; `fsim-nki` refuses an id no allocation of that arena declared.
- `arena` and `@addr` give the first byte, as in the allocation; `dtype` is the operand's dtype, a bitcast of the arena's bytes.
- The pattern is `[[stride, count], ...]`, outer dimension first.
- For `sbuf` and `psum` the first entry is the partition dimension, its stride in partitions, 0 for a broadcast.
- Every other entry's stride is in elements of `dtype` within the partition, or within HBM, 0 for a broadcast.
- The operand's shape is the counts.
- The recorder rewrites a view's element offset and pattern over its allocation's row into this form.
- It refuses a view whose partition stride is not a whole number of partitions.
- `+s(<view>)*<k>`, a run-time offset, adds the index the inner view reads, times `k` elements.
- `+r<N>*<k>` is reserved for the same with register `r<N>`, and `fsim-nki` refuses it.
- An index view is a single `i32` or `u32`, read as `i32`; a multi-element `+s(...)` view is refused.
- A `+s(...)` offset is lowered on a read view; a written view with one is refused.
- A `+s(...)` offset moves within a partition row, never across partitions.
- The recorder refuses a `+s(...)` offset on an SBUF or PSUM partition dimension.
- `dtype` is one of `bf16`, `f16`, `f32`, `i32`, `u8`, `u16`, `u32`.

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
`fsim-nki` refuses each of them:

| record                            | meaning                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| `reg.load r<N> <view>`            | submit, sync, read the view's first int32 into the register          |
| `reg.set r<N> <int>`              | set the register                                                     |
| `loop r<N> <start> <stop> <step>` | push a loop; the register counts from `start`                        |
| `endloop`                         | step the register and jump to the loop's first record while `< stop` |

### End of kernel

`output out %2` names a returned tensor; `fsim-nki` writes each sample's bytes to `<outputs>/out.bin` after its batch's sync.
`end` is the last line; a stream without it is an error.

### Example

The records of `rmsnorm_tkg/llama3_8b_tp4`, abridged, as the recorder emits them:

```
nki-trace 2 target=trn2 lnc=1 hbm=49152
alloc %0 hbm@0 bf16[1,1,4096] driver input=input
alloc %1 hbm@8192 bf16[1,4096] driver input=gamma
alloc %2 hbm@16384 bf16[128,1,32] driver name=out
alloc %3 sbuf@0,0 bf16[128,1,32] driver
dma_copy dst=sbuf%3@0,0:bf16[[1,128],[32,1],[32,1],[1,32]] src=hbm%0@0:bf16[[32,128],[4096,1],[4096,1],[1,32]] oob_is_err=true engine=dma
alloc %5 sbuf@0,128 f32[128,1] driver
memset dst=sbuf%5@0,128:f32[[1,128],[1,1]] value=1e-06 engine=unknown
alloc %6 sbuf@0,132 f32[128,128] driver
memset dst=sbuf%6@0,132:f32[[1,128],[1,128]] value=1.0 engine=unknown
alloc %7 sbuf@0,644 f32[128,1,32] driver
activation dst=sbuf%7@0,644:f32[[1,128],[32,1],[1,32]] op=square data=sbuf%3@0,0:bf16[[1,128],[32,1],[1,32]] bias=0.0 scale=1.0 reduce_op=none reduce_res=none reduce_cmd=idle
alloc %8 sbuf@0,772 f32[128,1] driver
tensor_reduce_arith dst=sbuf%8@0,772:f32[[1,128],[1,1]] op=add data=sbuf%7@0,644:f32[[1,128],[32,1],[1,32]] num_r_dim=1 negate=false keepdims=false
alloc %10 psum@0,0 f32[128,1] driver
nc_matmul dst=psum%10@0,0:f32[[1,128],[1,1]] stationary=sbuf%6@0,132:f32[[1,128],[1,128]] moving=sbuf%8@0,772:f32[[1,128],[1,1]] is_transpose=false row_pos=none col_pos=none perf_mode=none accumulate=none
output out %2
end
```

## Layout

The driver is a pipeline: `Parser` parses each line into a record, and `Driver` runs it over an `Accelerator` as it arrives.
Decode is one free function per instruction kind, which the decode table in `src/isa/Instruction.cpp` dispatches by the kind's name.
Each decode builds an `Instruction` whose `semantics(Accelerator&, Module&)` emits its StableHLO into a `Module`.
`main.cpp` owns the `Driver` and the `BatchLoop`, the `BatchLoop` owns fsim's `Runtime` and `Pipeline`, and the `Driver` runs the modules through it.
The `BatchLoop` calls the `Driver`, never the reverse: `Driver::trace` takes its input and upload callbacks.
Parsing, accelerator state, code generation and instruction semantics each stay in one layer, and each layer includes only those before it:

1. `src/trace/`
2. `src/accelerator/` and `src/stablehlo/`, each over `trace/` alone
3. `src/isa/`
4. `src/Driver.{h,cpp}`
5. `src/samples/` and `src/main.cpp`

| path                                  | role                                                                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `record/backend.py`                   | the backend `nki._backends._activate_backend` installs: tensor builtins to records, allocations to placement                |
| `record/place.py`                     | the per-arena first-fit placement for allocations without an address                                                        |
| `record/record.py`                    | `record(build, out)`: runs the kernel under the backend and writes the stream to a file or a pipe                           |
| `src/trace/View.h`                    | `View`, one operand, and `Dyn`, its `+s(...)` offset; `Arena`, `Dtype` and a pattern's `Entry`                              |
| `src/trace/Fields.h`                  | `Fields`: an instruction record's values, which its decode takes by name and `done()` checks are all taken                  |
| `src/trace/Kernel.{h,cpp}`            | `Kernel`: the header and the outputs; `Alloc` and `Output`                                                                  |
| `src/trace/Parser.{h,cpp}`            | one line to one record: tokenizer, view and pattern parser; it refuses the reserved records                                 |
| `src/isa/Instruction.h`               | `Instruction` and its `semantics(Accelerator&, Module&)`; `decoder(kind)` and the `decode<Kind>(Fields&)`                   |
| `src/isa/Instruction.cpp`             | the decode table from a kind to its decode; any other kind is refused as unknown                                            |
| `src/isa/<Instruction>.cpp`           | one kind's `Instruction` subclass with `const` fields, and its decode, named after its record kind                          |
| `src/isa/Shared.{h,cpp}`              | the lowering helpers several kinds share: `copyView`, `selectView`, `applyOperand`, `compares`, `scalarOp`, `scalarArith`   |
| `src/accelerator/Accelerator.{h,cpp}` | `Accelerator`: the batch's host copies of the arenas and registers, the allocation table, the PSUM written mask             |
| `src/accelerator/Limits.{h,cpp}`      | the memory-model checks: an allocation within its arena, `nc_matmul`'s arenas and its dst in one PSUM bank                  |
| `src/stablehlo/Module.{h,cpp}`        | `Module`: state parameters, view `read` and `write`, ops, and `text()`, which ends the module and returns its text          |
| `src/Driver.{h,cpp}`                  | `Driver`: `trace()` over a file or stdin, `validate()`, a parse-only reading for `+s(...)` offsets, `resolve`, soft trigger |
| `src/samples/BatchLoop.{h,cpp}`       | `BatchLoop`: the loop over a run's batches, replay, and a batch size of 1 for a trace with a `+s(...)` offset               |
| `src/samples/Inputs.{h,cpp}`          | `Inputs`: the case's input sets, `--sampling`, and a sample's load from its input set                                       |
| `src/samples/Outputs.{h,cpp}`         | each sample's outputs into `<outputs>/<name>.bin`, after the output cap and disk check, and `draws.bin`                     |
| `src/main.cpp`                        | the `fsim-nki` command line                                                                                                 |
| `CMakeLists.txt`                      | `add_subdirectory(../../fsim)`, links `fsim`                                                                                |

## From a record to StableHLO

The library runs a fixed set of state tensors through modules.
They are the driver's data models, in TAIDL's sense of a declared buffer an instruction addresses ([declarations](https://github.com/act-compiler/taidl/blob/v2.14/docs/declarations.md)).
`hbm` plays TAIDL's `d0`, the off-chip memory as a flat buffer of bytes.
The driver's state is five:

| state              | type               | what it is                                                                                                                                                                               |
| ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hbm`              | `ui8[1, hbm]`      | the HBM arena at the header's capacity                                                                                                                                                   |
| `sbuf`             | `ui8[128, 229376]` | SBUF, 224 KiB per partition                                                                                                                                                              |
| `psum`             | `ui8[128, 16384]`  | PSUM, 8 banks of 2 KiB per partition                                                                                                                                                     |
| `reduce_regs`      | `f32[128, 1]`      | the activation engine's reduce registers ([nki.isa.activation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.activation.html))                      |
| `accumulator_regs` | `f32[128, 1]`      | the Vector Engine's accumulator registers ([nki.isa.tensor_scalar_reduce](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_scalar_reduce.html)) |

- SBUF is modelled at the documented 224 KiB per partition.
- nki's `nl.tile_size.sbuf_fmax_bytes` gives 212984 usable bytes per partition on trn2, 16 KiB and 8 bytes below 224 KiB; the driver checks the documented bound.

One instruction is one `semantics` call, which reads its views, computes, and writes its destination view.
A view is an NKI access pattern over a region of one data model.
It lowers to the ops the [StableHLO spec](https://openxla.org/stablehlo/spec) defines.

- `read`: the arena is reshaped to a `[P, R/K, K]` byte grid and bitcast to the view's `K`-byte dtype, a `[P, R/K]` element grid.
- Unit counts drop out of the pattern, and zero strides fold to one element.
- A single-level pattern is one strided [slice](https://openxla.org/stablehlo/spec#slice) of the grid, its partitions and elements together.
- A nested pattern, each stride a multiple of the next and the entries disjoint, is a slice of its window first.
- The window is then delinearized into a reshape, one dimension per entry, as MLIR's [affine.delinearize_index](https://mlir.llvm.org/docs/Dialects/Affine/#affinedelinearize_index-affineaffinedelinearizeindexop) splits a flat index, and sliced to the counts.
- A pattern that nests only with its entries sorted by stride, outer first, is read in that order and transposed back to the view's.
- Any other pattern falls back to a gather with a constant index.
- A zero stride becomes a broadcast back to the full shape.
- A view with a `+s(...)` offset reads the index, clamps it to the range `Accelerator::resolve` finds, and computes the window's first byte from it.
  The range holds the index values that keep the view in its allocation.
- A one-index [gather](https://openxla.org/stablehlo/spec#gather) takes that window from the arena's bytes, every partition, and bitcasts it to the dtype grid.
- The paths above then read the view from the window as from the arena.
- The window spans what the path slices: the one-level slice, the nested window, or the gathered elements.
- A nested view's window stays in the arena row at every index of the range, or the view takes the gather path.
- An index in its range reads the bytes a constant address would; one outside it clamps, where a constant address past the allocation is refused.
- The result is `[count_0, ..., count_n]` in the view's dtype.
- compute: the ops the numpy simulator spells, over f32 converts of the operands.
- `write`: a nested pattern at partition stride 1 is a dynamic_update_slice of the arena's window, after an update of the delinearized window when the pattern leaves gaps.
- A pattern that nests only once sorted takes the value transposed into the sorted order first.
- Any other pattern falls back to a scatter; either way the grid is bitcast back to bytes.
- A write through a zero-stride view first slices the value to each zero-stride entry's last index, then writes the rest of the view.
- The library's byte-grid pass keeps bytes as bytes below the bitcasts.
- An f32 to integer convert is round-to-nearest-even, saturating, NaN to the maximum, as nki.simulate casts.

Control and state:

- `Driver::execute` asks `Pipeline::soft_trigger` for the module's op count before each instruction.
- When it fires, `Module::text` ends the module and the driver submits its text, and the module starts afresh.
- Under `overlap`, submitted modules compile on the library's threads while the driver keeps decoding.
- `Driver::resolve` has `Accelerator::resolve` check every view before the instruction's decode and semantics.
- It checks each view against:
  - its allocation
  - its arena
  - its dtype's alignment
- For a `+s(...)` offset, it finds the range of index values that keeps the view in its allocation, and checks the view at the first.
- The index stays on the device, so no trace needs a mid-trace `sync`, one before its `end`.
- `nc_matmul` resolves `accumulate` at trace time from a PSUM written mask in `Accelerator`, per allocation:
  - Each PSUM byte holds the allocation that last wrote it, so a new allocation starts unwritten.
  - Any write to PSUM, by a matmul or not, sets its bytes to the view's allocation.
  - `true` adds onto the tile.
  - `false` clears the mask over the destination's allocation, then overwrites.
  - `none` adds where the destination's allocation wrote the tile and overwrites elsewhere.
  - `none` with `is_transpose=true` always overwrites.
  - A write by another allocation at the same bytes overwrites them, since the bytes alias.
  - It matches nki.simulate except when aliased allocations interleave writes.
  - So `true`, or a non-transposing `none`, on an allocation whose bytes another overwrote is refused, until an `accumulate=false` write restarts it.
- The reduce register lanes index from the operand's partition start, and `reduce_res` reads them.
- `reduce_cmd=reset` zeroes them, `reset_reduce` stores the reduction, and `reduce` adds it on.
- `activation` refuses a `reduce_op` other than `add` with `reduce` or `reset_reduce`, since the ISA reduces by add only ([nki.isa.activation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.activation.html)).
- The suite reaches `reduce_cmd=reset_reduce` with a `reduce_res` in `activation` and `tensor_scalar_reduce`, through attention_cte and qkv_cte.
- `tensor_scalar_reduce` reduces into `accumulator_regs` from `data`'s partition start, and its `reduce_res` reads them from its own, as nki.simulate does.
- Each instruction runs as its record arrives, and `instructionCounter_` counts them for the summary's `trace`.
- After `end`, the driver ends the trace, prints the summary and submits; its batch loop syncs and stores the outputs.

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
- A dst with a `+s(...)` offset is refused at its write, and `--validate` counts it in `bank_unchecked`.
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
- `nc_matmul` with `accumulate=none` adds onto a PSUM tile that a `memset` or copy wrote, as nki.simulate does, when the write was through the same allocation; the docs call this undefined on trn2.
- Transcendentals are XLA's f32 ops, and the hardware engines may round differently, as the NKI simulator guide's [Numerical Precision](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/nki_simulator.html#numerical-precision) section says of nki.simulate.
- `reduce_cmd=reduce` and `reset_reduce` on `activation` reduce by add, the one `reduce_op` the ISA documents.
- `reduce_cmd=load_reduce` on `activation` follows nki's simulator, which leaves the reduce registers unchanged; it is to be checked on the chip.
- `activation` `op=silu` is `x / (1 + exp(-x))`, as nki.simulate spells it; the docs list silu over `[-inf, inf]` with no formula ([Supported Activation Functions](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/nki.api.shared.html#tbl-act-func)).
- `tensor_copy` is a bit copy when `src` and `dst` share a dtype, and casts through f32 otherwise, on every engine, as nki.simulate does.
- The [nki.isa.tensor_copy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_copy.html) docs give the bit copy only on GpSimd and Vector, and send Scalar through f32 every time.
- That f32 convert is lossless for bf16 and f32, apart from NaN payloads.
- It writes in flat order, as nki.simulate does, so `[128,1,64]` takes `[128,64]`.
- The docs bar GpSimd from PSUM; nki.simulate ignores `engine`, and so does the driver.
- `affine_select` compares `offset + channel_id * channel_multiplier` plus the pattern's index with 0 by `cmp_op`, then takes `on_true_tile` or `on_false_value` ([nki.isa.affine_select](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.affine_select.html)).
- `channel_id` counts from `dst`'s first partition, as nki.simulate counts it.
- It selects in f32 and converts to `dst`'s dtype for every dtype pair, as nki.simulate does; the docs cast through f32 only when the dtypes differ.
- The docs put `dst` and `on_true_tile` in SBUF, and the driver refuses any other arena.
- A write through a zero-stride view keeps, per element, the value last in flat order, as nki.simulate leaves it.
- The suite reaches such a write in qkv_cte's `activation` into a scratch tile.
- `tensor_scalar_arith` computes `data op0 operand0` in f32 and casts to `dst`'s dtype ([nki.isa.tensor_scalar](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_scalar.html)).
- Its view `operand0` is unit-padded to the rank of `data`, or of `dst` when `data` and `dst` have equal counts, and broadcast, as nki.simulate's `broadcast_operand` does; the docs give it as `[P, 1]`.
- It ignores `engine`, as nki.simulate does.
- `dma_transpose` writes `src` permuted by `axes` to `dst` ([nki.isa.dma_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.dma_transpose.html)), in flat order as nki.simulate's `np.transpose` does.
- It does not check the docs' limits on `src.shape[0]`, which the suite's tiles exceed; nki.simulate does not check them either.
- `tensor_scalar_reduce` computes `data op0 operand0` as `tensor_scalar_arith` does, then reduces `dst` over its free dims by `reduce_op` ([nki.isa.tensor_scalar_reduce](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_scalar_reduce.html)).
- It reduces `dst` as written, in `dst`'s dtype then f32, as nki.simulate reads it back; `reset_reduce` stores that reduction, as a reset to the identity then a fold would.
- `reciprocal` computes `1.0 / data` in f32 and casts to `dst`'s dtype ([nki.isa.reciprocal](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.reciprocal.html)), writing in flat order as nki.simulate does.
- `tensor_tensor_scan` computes `r[i] = op1(op0(data0[i], r[i-1]), data1[i])` along the free dim per partition, from `initial`, in f32 ([nki.isa.tensor_tensor_scan](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_tensor_scan.html)).
- For multiply then add, it composes the maps `r -> data0[i] * r + data1[i]` in a Hillis-Steele scan, so its sums run in a tree order, not nki.simulate's loop order.
- The scan forms prefix products of `data0` before `initial` enters, so a product past f32's range gives a result the loop does not.
- `select_reduce` takes `on_true` where `predicate` is nonzero and `on_false` elsewhere, in f32, and casts to `dst`'s dtype ([nki.isa.select_reduce](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.select_reduce.html)).
- With `reduce_cmd=idle`, `select_reduce` leaves `accumulator_regs` unchanged, as nki.simulate does; the docs say idle may still modify the accumulator.
- `nc_stream_shuffle` gives each `dst` partition `p` the `src` partition `32 * (p / 32) + shuffle_mask[p % 32]`, the quadrant rule of [nki.isa.nc_stream_shuffle](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_stream_shuffle.html).
- It copies bits, as `tensor_copy` does, departing from nki.simulate's f32 convert, which would round a u32 above 2^24.
- `stream_transpose` is the record kind of [nki.isa.nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_transpose.html) on the Vector Engine.
- It writes `data` as `[P, free]` transposed to `dst` in flat order, as nki.simulate does.
- It refuses a `data` past 32x32, the Vector Engine bound of nki.isa.nc_transpose.
- `local_gather` gives each 16-partition GpSimd core's `dst` column `j` the `src_buffer` column of the core's `j`-th `index` entry, as nki.simulate does.
- The core's rows of `index` flatten partition-first, and the gather runs on the device, with no sync.
- It clips an `index` entry to the row's last column, as nki.simulate does; the 2.32 docs leave an out-of-range entry undefined.

## Coverage

The suite's thirteen kernels reach seventeen instruction kinds at trn2, each with one keyword set, no register load and no register loop.

| kind                   | kernels                                                 |
| ---------------------- | ------------------------------------------------------- |
| `dma_copy`             | all                                                     |
| `memset`               | all but rope and output_proj                            |
| `activation`           | norms, qkv, mlp, attention, moe                         |
| `tensor_tensor_arith`  | norms, rope, qkv_tkg, mlp, attention, moe               |
| `tensor_reduce_arith`  | norms, qkv_tkg, attention, moe all_expert               |
| `nc_matmul`            | all but rope and cumsum                                 |
| `tensor_copy`          | rope, cumsum, qkv_cte, output_proj, mlp, attention, moe |
| `tensor_tensor_scan`   | cumsum                                                  |
| `tensor_scalar_arith`  | qkv_cte, attention, moe                                 |
| `affine_select`        | qkv_cte, mlp, attention, moe                            |
| `dma_transpose`        | attention_cte                                           |
| `reciprocal`           | attention                                               |
| `tensor_scalar_reduce` | attention_cte                                           |
| `select_reduce`        | attention_tkg                                           |
| `nc_stream_shuffle`    | attention_tkg, moe                                      |
| `stream_transpose`     | attention_tkg, moe selective                            |
| `local_gather`         | moe selective                                           |

`fsim-nki` decodes these kinds; any other fails at decode as unknown.
Within a decoded kind, it decodes these values, and any other fails at decode, or at lowering where a row says so:

| field                                      | decodes                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------ |
| `activation` `op`                          | copy, exp, log, sqrt, rsqrt, tanh, sigmoid, silu, relu, square, abs, reciprocal      |
| `activation` `reduce_op`                   | add; with `reset` or `load_reduce`, also multiply, maximum, minimum                  |
| `tensor_tensor_arith` `op`                 | the arithmetic ops: add, subtract, multiply, divide, maximum, minimum, power         |
| `tensor_tensor_arith` `op`, compare        | the six `affine_select` compares, 1.0 or 0.0 in f32; the suite reaches equal         |
| `tensor_tensor_arith` `op`, gpsimd integer | add, subtract, multiply, maximum, minimum; compares refused                          |
| `tensor_reduce_arith` `op`                 | add, multiply, maximum, minimum                                                      |
| `nc_matmul` `perf_mode`                    | none                                                                                 |
| `tensor_copy` `engine`                     | any; the suite reaches vector, scalar, unknown                                       |
| `tensor_copy` `src` to `dst` dtype         | any pair; the suite reaches equal, f32 to bf16, u32 to f32, i32 to u16               |
| `tensor_scalar_arith` `op0`                | add, multiply, maximum                                                               |
| `tensor_scalar_arith` `operand0`           | a view, unit-padded to `data`'s rank, `dst`'s at equal counts; or any number, as f32 |
| `tensor_scalar_arith` `op1`, `operand1`    | none, with `reverse0` and `reverse1` false                                           |
| `tensor_scalar_arith` `engine`             | any; the suite reaches vector, unknown                                               |
| `affine_select` `cmp_op`                   | equal, not_equal, greater, greater_equal, less, less_equal                           |
| `affine_select` `channel_multiplier`       | any integer; the suite reaches 1, -1                                                 |
| `affine_select` `on_false_value`           | any number, as f32; the suite reaches -3.4028235e+38, 1.0                            |
| `affine_select` `pred`                     | any pattern of `dst`'s elements per partition; another count fails at lowering       |
| `dma_transpose` `axes`                     | [1, 0], [2, 1, 0], [3, 1, 2, 0], as documented; the suite reaches the last           |
| `dma_transpose` `src`, `dst`               | equal 2-byte dtypes, `src` in hbm or sbuf, `dst` in sbuf; the suite uses bf16        |
| `tensor_scalar_reduce` `op0`, `operand0`   | add, multiply, maximum, by a view or any number, as f32, with `reverse0` false       |
| `tensor_scalar_reduce` `reduce_op`         | add, multiply, maximum, minimum, as documented; the suite reaches maximum            |
| `tensor_scalar_reduce` `reduce_cmd`        | reset_reduce, with `reduce_init` none                                                |
| `tensor_scalar_reduce` `reduce_res`        | a view or none; the suite reaches an sbuf f32 view                                   |
| `reciprocal` `data` to `dst` dtype         | any pair, through f32; the suite reaches f32 to f32                                  |
| `tensor_tensor_scan` `op0`, `op1`          | multiply then add, with `reverse0` and `reverse1` false                              |
| `tensor_tensor_scan` `initial`             | a view, a scalar refused; the suite reaches an sbuf f32 `[128, 1]`                   |
| `select_reduce` `predicate`                | a u8 or u16 view, others refused, nonzero true; the suite reaches sbuf u8            |
| `select_reduce` `on_false`                 | any number, as f32, a view refused; the suite reaches -inf                           |
| `select_reduce` `reduce_cmd`, `reduce_op`  | idle with maximum, `reduce_res` none and `reverse_pred` false                        |
| `nc_stream_shuffle` `shuffle_mask`         | 32 equal entries below 32; the suite reaches all 0                                   |
| `nc_stream_shuffle` `src`, `dst`           | one dtype, per partition alike, at lowering; the suite reaches f32, bf16, u32        |
| `stream_transpose` `engine`                | vector, as nki.simulate requires                                                     |
| `stream_transpose` `data` to `dst` dtype   | equal at lowering; the suite reaches f32 `[32, 1]` and `[8, 1]`, u16 `[16, 16]`      |
| `local_gather` `num_elem_per_idx`          | 1                                                                                    |
| `local_gather` `num_valid_indices`         | `dst`'s elements per partition; the suite reaches 256                                |
| `local_gather` `index`                     | u16, others refused; the suite reaches sbuf `[128, 16]`                              |
| `local_gather` `index` shape               | `dst`'s partitions; per partition, at least a sixteenth of `dst`'s elements          |
| `local_gather` `src_buffer`, `dst`         | one dtype at lowering, partitions in whole GpSimd cores; the suite reaches f32       |

## Scope and limits

- One target, trn2 at lnc=1: one core, no `program_id` sharding, no `sendrecv` or `core_barrier`.
- The dtypes above, with bf16, f16, f32, i32 and u8 as the kernels' data; no fp8, no MX and no collectives, on which the backend raises.
- Execution is one program order, as in nki's simulator: no engines, semaphores or scheduling, so no race is modelled.
- Inputs are `shared_hbm`, as a compiled kernel sees them; this is what lets all_expert record.
- A kernel that converts a register to a Python value cannot be recorded; the backend raises, as nki's compiler frontend does.

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
- [nki.isa.tensor_scalar_reduce, Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.tensor_scalar_reduce.html): accumulator registers, `reduce_cmd` and `reduce_res`.
