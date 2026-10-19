# The pass stage over the static MLIR toolchain of ../setup.sh.
# fsim_passes is a shared object exporting only passes.h's C entry, so its
# LLVM never meets a driver's; fsim-passes runs the stage over a file.
set(FSIM_MLIR_DIR "${PROJECT_SOURCE_DIR}/third_party/mlir")
if(NOT EXISTS "${FSIM_MLIR_DIR}/lib")
  message(FATAL_ERROR "No MLIR toolchain in ${FSIM_MLIR_DIR}: run ./scripts/setup.sh --mlir")
endif()
file(GLOB FSIM_MLIR_LIBS "${FSIM_MLIR_DIR}/lib/*.a")

# A driver's include dirs and definitions, e.g. a system LLVM's, stay out.
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "" COMPILE_DEFINITIONS "")

# The toolchain: headers, no-RTTI flags and every library in one group.
add_library(fsim_mlir INTERFACE)
target_include_directories(fsim_mlir SYSTEM INTERFACE "${FSIM_MLIR_DIR}/include")
target_compile_definitions(fsim_mlir INTERFACE LLVM_BUILD_STATIC _GNU_SOURCE __STDC_CONSTANT_MACROS
  __STDC_FORMAT_MACROS __STDC_LIMIT_MACROS)
target_compile_options(fsim_mlir INTERFACE -fno-rtti)
target_link_libraries(fsim_mlir INTERFACE -Wl,--start-group ${FSIM_MLIR_LIBS} -Wl,--end-group Threads::Threads
  ${CMAKE_DL_LIBS})

# The stage's objects, shared by the shared object and the tool.
add_library(fsim_passes_stage OBJECT passes.cc fsim_passes.cc window.cc byte_grids.cc forward_slices.cc
  dead_updates.cc)
set_target_properties(fsim_passes_stage PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(fsim_passes_stage PUBLIC cxx_std_20)
target_compile_options(fsim_passes_stage PRIVATE -Wall -Wextra -Werror -fvisibility=hidden
  -fvisibility-inlines-hidden)
target_link_libraries(fsim_passes_stage PUBLIC fsim_mlir)

# Only the C entry leaves the object, its static libraries hidden.
add_library(fsim_passes SHARED)
target_link_libraries(fsim_passes PRIVATE fsim_passes_stage)
target_link_options(fsim_passes PRIVATE -Wl,--exclude-libs,ALL -Wl,-z,defs
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
set_target_properties(fsim_passes PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
target_include_directories(fsim_passes INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# The tool sits beside the library's build output.
add_executable(fsim-passes tool.cc)
target_compile_options(fsim-passes PRIVATE -Wall -Wextra -Werror)
target_link_libraries(fsim-passes PRIVATE fsim_passes_stage)
set_target_properties(fsim-passes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
