# fsim-lib

fsim-lib is a scalable functional simulator for AI accelerators.
It is a common simulation library that ISA drivers sit over, each handing the library its ISA's tensor instructions.
The library builds StableHLO lazily and compiles and runs it through XLA's PJRT C API.

Under review at a major systems conference.

## Layout

- `fsim/`: the library; `fsim/README.md` documents it.
