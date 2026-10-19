# The fsim library: a static target, linked by consumers as fsim.
cmake_minimum_required(VERSION 3.20)
project(fsim LANGUAGES CXX)

# The build's variant and compiler, each fixed at configure time.
set(FSIM_VARIANT overlap CACHE STRING "Variant: eager, taidl, trigger or overlap")
if(NOT FSIM_VARIANT MATCHES "^(eager|taidl|trigger|overlap)$")
  message(FATAL_ERROR "Unknown FSIM_VARIANT '${FSIM_VARIANT}': pass eager, taidl, trigger or overlap")
endif()
set(FSIM_COMPILER custom-cpu CACHE STRING "Compiler: xla-0.11-cpu, xla-0.4-cpu or custom-cpu")
# The compiler picks the PJRT entry of fsim/setup.sh.
if(FSIM_COMPILER MATCHES "^(xla-0\\.11|custom)-cpu$")
  set(FSIM_PJRT jax-0.11.2-cpu)
elseif(FSIM_COMPILER STREQUAL "xla-0.4-cpu")
  set(FSIM_PJRT jax-0.4.33-cpu)
else()
  message(FATAL_ERROR "Unknown FSIM_COMPILER '${FSIM_COMPILER}': pass xla-0.11-cpu, xla-0.4-cpu or custom-cpu")
endif()
set(FSIM_PJRT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt/${FSIM_PJRT}")
# The plugin's absolute path, baked into the library.
set(FSIM_PJRT_PLUGIN "${FSIM_PJRT_DIR}/lib/pjrt_c_api_cpu_plugin.so")
# A missing entry stops the configure.
if(NOT EXISTS "${FSIM_PJRT_PLUGIN}")
  message(FATAL_ERROR "No ${FSIM_PJRT_PLUGIN}: run ./scripts/setup.sh --pjrt ${FSIM_PJRT}")
endif()
string(TOUPPER "${FSIM_VARIANT}" FSIM_VARIANT_NAME)
# The compiler's kind, xla or custom, names its definition.
string(REGEX MATCH "^[a-z]+" FSIM_COMPILER_NAME "${FSIM_COMPILER}")
string(TOUPPER "${FSIM_COMPILER_NAME}" FSIM_COMPILER_NAME)

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
  FSIM_VARIANT_${FSIM_VARIANT_NAME} FSIM_COMPILER_${FSIM_COMPILER_NAME})
target_compile_options(fsim PRIVATE -Wall -Wextra -Werror)
target_link_libraries(fsim PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
