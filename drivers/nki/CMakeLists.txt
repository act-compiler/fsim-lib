# fsim-nki: the NKI driver over the fsim library, at its FSIM_VARIANT and FSIM_COMPILER.
cmake_minimum_required(VERSION 3.20)
project(fsim_nki LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../fsim fsim)

add_executable(fsim-nki
  src/main.cpp
  src/Driver.cpp
  src/accelerator/Accelerator.cpp
  src/accelerator/Limits.cpp
  src/isa/Activation.cpp
  src/isa/DmaCopy.cpp
  src/isa/Instruction.cpp
  src/isa/Memset.cpp
  src/isa/NcMatmul.cpp
  src/isa/TensorReduce.cpp
  src/isa/TensorTensor.cpp
  src/stablehlo/Module.cpp
  src/trace/Kernel.cpp
  src/trace/Parser.cpp
)
target_compile_features(fsim-nki PRIVATE cxx_std_20)
target_include_directories(fsim-nki PRIVATE src)
target_compile_options(fsim-nki PRIVATE -Wall -Wextra -Werror)
target_link_libraries(fsim-nki PRIVATE fsim)
