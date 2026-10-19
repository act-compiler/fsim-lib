# The fsim library: a static target, linked by consumers as fsim.
cmake_minimum_required(VERSION 3.20)
project(fsim LANGUAGES CXX)

# The build's variant and compiler, each fixed at configure time.
set(FSIM_VARIANT overlap CACHE STRING "Variant: eager, taidl, trigger or overlap")
if(NOT FSIM_VARIANT MATCHES "^(eager|taidl|trigger|overlap)$")
  message(FATAL_ERROR "Unknown FSIM_VARIANT '${FSIM_VARIANT}': pass eager, taidl, trigger or overlap")
endif()
set(FSIM_COMPILER custom-cpu CACHE STRING
  "Compiler: xla-0.11-cpu, xla-0.4-cpu, custom-cpu, xla-0.11-gpu, xla-0.4-gpu or custom-gpu")
# The compiler picks the PJRT entry of fsim/setup.sh.
if(FSIM_COMPILER MATCHES "^(xla-0\\.11|custom)-cpu$")
  set(FSIM_PJRT jax-0.11.2-cpu)
elseif(FSIM_COMPILER STREQUAL "xla-0.4-cpu")
  set(FSIM_PJRT jax-0.4.33-cpu)
elseif(FSIM_COMPILER MATCHES "^(xla-0\\.11|custom)-gpu$")
  set(FSIM_PJRT jax-0.11.2-cuda12)
  # Its CUDA libraries under nvidia/, in jax's preload order.
  set(FSIM_PJRT_CUDA cuda_runtime/lib/libcudart.so.12 cuda_nvrtc/lib/libnvrtc.so.12 cublas/lib/libcublas.so.12
    cublas/lib/libcublasLt.so.12 nccl/lib/libnccl.so.2 cuda_cupti/lib/libcupti.so.12 cusparse/lib/libcusparse.so.12
    cusolver/lib/libcusolver.so.11 cufft/lib/libcufft.so.11 nvshmem/lib/libnvshmem_host.so.3 cudnn/lib/libcudnn.so.9)
elseif(FSIM_COMPILER STREQUAL "xla-0.4-gpu")
  set(FSIM_PJRT jax-0.4.33-cuda12)
else()
  message(FATAL_ERROR "Unknown FSIM_COMPILER '${FSIM_COMPILER}': pass xla-0.11-cpu, xla-0.4-cpu, custom-cpu, "
    "xla-0.11-gpu, xla-0.4-gpu or custom-gpu")
endif()
set(FSIM_PJRT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt/${FSIM_PJRT}")
# The plugin's absolute path, baked into the library.
if(FSIM_PJRT MATCHES "-cuda12$")
  set(FSIM_PJRT_PLUGIN "${FSIM_PJRT_DIR}/lib/xla_cuda_plugin.so")
else()
  set(FSIM_PJRT_PLUGIN "${FSIM_PJRT_DIR}/lib/pjrt_c_api_cpu_plugin.so")
endif()
# A missing entry stops the configure.
if(NOT EXISTS "${FSIM_PJRT_PLUGIN}")
  message(FATAL_ERROR "No ${FSIM_PJRT_PLUGIN}: run ./scripts/setup.sh --pjrt ${FSIM_PJRT}")
endif()
string(TOUPPER "${FSIM_VARIANT}" FSIM_VARIANT_NAME)
# The compiler's kind, xla or custom, names its definition.
string(REGEX MATCH "^[a-z]+" FSIM_COMPILER_NAME "${FSIM_COMPILER}")
string(TOUPPER "${FSIM_COMPILER_NAME}" FSIM_COMPILER_NAME)
# The compiler's device, cpu or gpu, names its definition.
string(REGEX MATCH "[a-z]+$" FSIM_DEVICE_NAME "${FSIM_COMPILER}")
string(TOUPPER "${FSIM_DEVICE_NAME}" FSIM_DEVICE_NAME)

find_package(Threads REQUIRED)

add_library(fsim STATIC
  src/compile_options.cc
  src/pipeline.cc
  src/runtime.cc
  src/tensor.cc
)
target_compile_features(fsim PUBLIC cxx_std_20)
target_include_directories(fsim PUBLIC include PRIVATE src)
target_include_directories(fsim SYSTEM PRIVATE "${FSIM_PJRT_DIR}/include")
target_compile_definitions(fsim PRIVATE "FSIM_PJRT_PLUGIN=\"${FSIM_PJRT_PLUGIN}\""
  FSIM_VARIANT_${FSIM_VARIANT_NAME} FSIM_COMPILER_${FSIM_COMPILER_NAME} FSIM_DEVICE_${FSIM_DEVICE_NAME})
# The entry's nvidia/ dir and its CUDA libraries, as C strings.
if(FSIM_PJRT_CUDA)
  list(TRANSFORM FSIM_PJRT_CUDA REPLACE "^.+$" "\"\\0\"")
  list(JOIN FSIM_PJRT_CUDA "," FSIM_PJRT_CUDA)
  target_compile_definitions(fsim PRIVATE "FSIM_PJRT_NVIDIA=\"${FSIM_PJRT_DIR}/nvidia\""
    "FSIM_PJRT_CUDA=${FSIM_PJRT_CUDA}")
endif()
target_compile_options(fsim PRIVATE -Wall -Wextra -Werror)
target_link_libraries(fsim PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
