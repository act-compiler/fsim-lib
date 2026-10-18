# The fsim library: a static target, linked by consumers as fsim.
cmake_minimum_required(VERSION 3.20)
project(fsim LANGUAGES CXX)

set(FSIM_PJRT_PLUGIN "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt/lib/libpjrt_cpu.so"
    CACHE FILEPATH "Absolute path of the PJRT plugin, baked into the library")

find_package(Threads REQUIRED)

add_library(fsim STATIC
  src/compile_options.cc
  src/pipeline.cc
  src/runtime.cc
  src/tensor.cc
)
target_compile_features(fsim PUBLIC cxx_std_20)
target_include_directories(fsim PUBLIC include PRIVATE src)
target_include_directories(fsim SYSTEM PRIVATE third_party/pjrt/include)
target_compile_definitions(fsim PRIVATE "FSIM_PJRT_PLUGIN=\"${FSIM_PJRT_PLUGIN}\"")
target_compile_options(fsim PRIVATE -Wall -Wextra -Werror)
target_link_libraries(fsim PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
