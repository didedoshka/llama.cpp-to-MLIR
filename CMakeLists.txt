cmake_minimum_required(VERSION 3.21)

project(ggml-mlir C CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add llvm and mlir
find_package(MLIR REQUIRED CONFIG)
message("LLVM_DIR ${LLVM_DIR}")

# important: -debug flag doesn't work without it
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})

# for files generated from tablegen
include_directories(${CMAKE_BINARY_DIR}/source/MLIRGGML)

# add llama
add_subdirectory(llama.cpp)

# add my libraries and executables
add_subdirectory(source)
