# The fsim library: a static target, linked by consumers as fsim.
cmake_minimum_required(VERSION 3.20)
project(fsim LANGUAGES CXX)

set(FSIM_PJRT jax-0.11.2 CACHE STRING "PJRT entry of fsim/setup.sh: header and plugin")
set(FSIM_PJRT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt/${FSIM_PJRT}")
# The plugin's absolute path, baked into the library.
set(FSIM_PJRT_PLUGIN "${FSIM_PJRT_DIR}/lib/pjrt_c_api_cpu_plugin.so")

# The build's variant and compiler, each fixed at configure time.
set(FSIM_VARIANT overlap CACHE STRING "Variant: eager, taidl, trigger or overlap")
set_property(CACHE FSIM_VARIANT PROPERTY STRINGS eager taidl trigger overlap)
if(NOT FSIM_VARIANT MATCHES "^(eager|taidl|trigger|overlap)$")
  message(FATAL_ERROR "Unknown FSIM_VARIANT '${FSIM_VARIANT}': pass eager, taidl, trigger or overlap")
endif()
set(FSIM_COMPILER custom CACHE STRING "Compiler: xla or custom")
set_property(CACHE FSIM_COMPILER PROPERTY STRINGS xla custom)
if(NOT FSIM_COMPILER MATCHES "^(xla|custom)$")
  message(FATAL_ERROR "Unknown FSIM_COMPILER '${FSIM_COMPILER}': pass xla or custom")
endif()
string(TOUPPER "${FSIM_VARIANT}" FSIM_VARIANT_NAME)
string(TOUPPER "${FSIM_COMPILER}" FSIM_COMPILER_NAME)

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
