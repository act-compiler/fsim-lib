# nki.simulate and the chip

This is an AI-generated report.
It compares `nki.simulate` with a trn2 chip through black-box runs, for comparative evaluation.

`nki.simulate` is NKI's CPU simulator, described in the [NKI simulator guide for Neuron SDK 2.32](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/guides/nki_simulator.rst).
Each finding pairs a kernel under `nki.simulate` on an x86 host with the same kernel on the chip.
The setup is under [Reproduction](#reproduction).

## The suite on the chip

- All 56 nkilib cases of `benchmarks/nki/` pass against their torch goldens on the chip.
- The worst relative error is 7.75e-3, within the bf16 tolerance of 3e-2.
- That includes nkilib's moe_tkg all_expert, which `nki.simulate` rejects (see [Kernel input placement](#kernel-input-placement)).

## Documented limitations

The 2.32 guide's Simulation Limitations section lists these limitations.

| Limitation          | The guide's words                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Independent tensors | "The simulator allocates each tensor independently without simulating overlapping memory regions" |
| No capacity check   | "or validating against SBUF/PSUM capacity limits"                                                 |
| No engine model     | "It does not model instruction latency, engine parallelism, or hardware scheduling."              |
| Partial validation  | "Some hardware constraint checks are missing"                                                     |
| No collectives      | "`nki.collectives` APIs are not implemented in the simulator."                                    |

## Memory model

`nki.simulate` evaluates each instruction's numerical semantics on per-tensor NumPy arrays and, per its guide, abstracts the memory model away.
This repo's driver models memory as the NKI documentation describes it, with each tensor at the address the kernel gives it.
HBM is byte-addressed, and SBUF and PSUM are partitioned, byte-addressed memories at their documented sizes, with PSUM in 2 KiB banks.
Kernel inputs sit in shared HBM, as on the chip (see the next section).

| Aspect                        | `nki.simulate`                                  | This repo's driver                                             | Public source                                                                                                                                                                                                                                                                                                    |
| ----------------------------- | ----------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensor placement              | Each tensor is its own array, with no address   | Each tensor is a region of one memory, at the kernel's address | [memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/get-started/about/memory-hierarchy-overview.html#neuroncore-internal-memory)                                                                                                                                                     |
| SBUF and PSUM capacity        | Not checked, per the guide                      | SBUF at 224 KiB and PSUM at 16 KiB per partition, checked      | [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium2_arch.html#neuroncore-v3-compute-engine-updates), [PSUM](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html#near-memory-accumulation-in-psum) |
| Partition range               | Checked on tile shape, not on a start partition | Partitions 0 to 127, checked at the kernel's address           | [memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/get-started/about/memory-hierarchy-overview.html#neuroncore-internal-memory)                                                                                                                                                     |
| PSUM banks for `nc_matmul`    | No banks                                        | Each `nc_matmul` dst within one 2 KiB bank, checked            | [PSUM](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html#near-memory-accumulation-in-psum), [nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.isa.nc_matmul.html)                                         |
| Dtype views of the same bytes | Each array holds its own dtype                  | A dtype is a view, so two dtypes read the same bytes           | [NkiTensor.view](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/api/generated/nki.language.NkiTensor.html)                                                                                                                                                                                         |
| Engine parallelism and timing | Not modelled                                    | Not modelled: one program order, no timing                     | [simulator guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/nki_simulator.html#performance)                                                                                                                                                                                            |

The `xfail` cases of `benchmarks/nki/` hold one small kernel per limit, each breaking it at an explicit address.

| Case                                  | nki's compile path                                                            | `nki.simulate` | This repo's driver                                    |
| ------------------------------------- | ----------------------------------------------------------------------------- | -------------- | ----------------------------------------------------- |
| SBUF capacity, bytes 229120 to 229631 | rejected: "Allocated memory out of bound"                                     | runs           | rejected: past the 229376 bytes per partition of sbuf |
| Partitions 64 to 191                  | rejected: "partition constraint solver failed"                                | runs           | rejected: past the 128 partitions of sbuf             |
| PSUM capacity, bytes 14336 to 18431   | rejected: "Allocated memory out of bound"                                     | runs           | rejected: past the 16384 bytes per partition of psum  |
| `nc_matmul` dst across a PSUM bank    | rejected: "Matmult cannot write to multiple PSUM banks for architecture Trn2" | runs           | rejected: across a 2048-byte PSUM bank boundary       |

Build the driver, then generate and run the cases:

```bash
./scripts/build.sh nki
./scripts/bench.sh gen --isa nki --bench xfail
./scripts/bench.sh run --isa nki --bench xfail
```

- The gen log gives the compile and `nki.simulate` columns, one line per case.
- The run gives the driver's column, with each case `xfail` when the driver rejects it as expected.

The checks have these limits:

- They are exact only for tensors the kernel places at an explicit address.
- Tensors without an explicit address are checked at the driver's own placement, since the recorder sees no address for these tensors.
- nkilib's allocator passes an address unless it runs in its auto-allocation mode ([nki-library at 92d11f6, allocator.py](https://github.com/aws-neuron/nki-library/blob/92d11f6/src/nkilib_src/nkilib/core/utils/allocator.py#L412-L421)).
- SBUF is modelled at the documented 224 KiB per partition.
- nki's compile path rejects some SBUF ranges below 224 KiB; the docs give no usable size.
- No claim is made about overlapping or aliased tensors.

## Kernel input placement

The guide's limitations section does not list this difference.
The guide says that under `nki.simulate` "each NumPy array argument is wrapped into an `NkiTensor` with `buffer=nl.hbm`" at LNC=1.
So simulate places kernel inputs in private HBM, while the chip run gives the same kernel its inputs in shared HBM.
A kernel that branches on the buffer kind therefore takes a different path in simulation.

```python
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def kernel(x):
    out = nl.ndarray((128, 64), nl.float32, buffer=nl.shared_hbm)
    o = nl.ndarray((128, 64), nl.float32, buffer=nl.sbuf)
    if x.buffer == nl.shared_hbm:
        nisa.memset(dst=o, value=1.0)
    else:
        nisa.memset(dst=o, value=2.0)
    nisa.dma_copy(dst=out, src=o)
    return out


x = np.zeros((128, 64), np.float32)
print(nki.simulate(kernel)(x)[0, 0])
```

| Run                  | Output                   |
| -------------------- | ------------------------ |
| `nki.simulate`, trn2 | 2.0 in all 8192 elements |
| chip                 | 1.0 in all 8192 elements |

nkilib's moe_tkg all_expert is incompatible with simulate's input placement.
Its affinity masking loads `expert_index` into SBUF only when `expert_index.buffer == nl.shared_hbm` ([nki-library at 92d11f6, moe_tkg_affinity_masking.py](https://github.com/aws-neuron/nki-library/blob/92d11f6/src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg_affinity_masking.py#L229-L234)).

| Run                                  | Result                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `nki.simulate`, inputs as given      | rejected: `tensor_tensor data1 must be in ['sbuf', 'psum'], got private_hbm` |
| `nki.simulate`, inputs in shared HBM | PASS, relative error 6.17e-3                                                 |
| chip                                 | PASS                                                                         |

## Speed

`nki.simulate` wall times at target trn2, on one AMD EPYC 9124 socket with 16 cores, 32 threads and 62 GB RAM.
Each run uses `nki.simulate` as shipped, with default threading and numpy on OpenBLAS 0.3.34.

| Case                                     | Time     | Result                                                                    |
| ---------------------------------------- | -------- | ------------------------------------------------------------------------- |
| cumsum llama3_8b_tp4                     | 12.77 s  | PASS                                                                      |
| rope llama3_8b_tp4                       | 0.61 s   | PASS                                                                      |
| rmsnorm_tkg llama3_70b_tp8               | 0.00 s   | PASS                                                                      |
| layernorm_tkg llama3_70b_tp8             | 0.00 s   | PASS                                                                      |
| moe_tkg selective llama4_scout_tp8       | 4.68 s   | PASS                                                                      |
| moe_tkg all_expert llama4_scout_tp8      | -        | ERROR: `tensor_tensor data1 must be in ['sbuf', 'psum'], got private_hbm` |
| moe_tkg all_expert, inputs in shared HBM | 3.97 s   | PASS                                                                      |
| qkv_tkg llama3_70b_tp8_batch1            | 0.22 s   | PASS                                                                      |
| output_proj_tkg llama3_70b_tp8_batch1    | 0.12 s   | PASS                                                                      |
| mlp_tkg llama3_70b_tp8_batch1            | 1.51 s   | PASS                                                                      |
| attention_tkg batch1 kv8192              | 0.14 s   | PASS                                                                      |
| attention_tkg batch64 kv2048             | 2.27 s   | PASS                                                                      |
| qkv_cte seq512                           | 0.87 s   | PASS                                                                      |
| qkv_cte seq8192                          | 13.46 s  | PASS                                                                      |
| output_proj_cte seq512                   | 0.49 s   | PASS                                                                      |
| output_proj_cte seq8192                  | 7.51 s   | PASS                                                                      |
| mlp_cte seq512                           | 7.50 s   | PASS                                                                      |
| mlp_cte seq8192                          | 108.06 s | PASS                                                                      |
| attention_cte seq512                     | 0.30 s   | PASS                                                                      |
| attention_cte seq8192                    | 28.67 s  | PASS                                                                      |

This table is the baseline the driver is compared against.

## Reproduction

| Item                      | Version         |
| ------------------------- | --------------- |
| nki                       | 0.6.0           |
| neuronx-cc                | 2.27.5334.0     |
| Simulator guide           | Neuron SDK 2.32 |
| Instance family           | trn2            |
| Logical NeuronCore config | LNC=1           |
| numpy BLAS                | OpenBLAS 0.3.34 |

- Simulate runs on an x86 host in this repo's container, with the target set to trn2.
- The chip side compiles the same kernel for trn2 with nki's compile API and runs it through nki's public runtime, `nki.runtime`, on one NeuronCore.
- The suite's inputs and torch goldens come from nkilib's torch references, through this repo's generator.

Generate the suite cases:

```bash
./docker/exec.sh "benchmarks/nki/gen/regen.sh"
```

`repro.py` stands for the reproducer above, with a host side that calls `nki.simulate` and an instance side that compiles and runs the kernel.

Run it under `nki.simulate` at trn2 inside the container:

```bash
./docker/exec.sh "NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -I repro.py"
```

Run it on the chip, at LNC=1 on one NeuronCore of a trn2 instance:

```bash
export NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0
python repro.py
```
