cmake_minimum_required(VERSION 3.21)

project(ggml-mlir C CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# add llama
# include(FetchContent)
# FetchContent_Declare(
#   llama
#   GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
#   GIT_TAG master)
# FetchContent_MakeAvailable(llama)

add_subdirectory(llama.cpp)

# # add llvm and mlir find_package(MLIR REQUIRED CONFIG)
#
# message("LLVM_DIR ${LLVM_DIR}")
#
# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
#
# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}") list(APPEND
# CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}") include(TableGen) include(AddLLVM)
# include(AddMLIR) include(HandleLLVMOptions)
#
# include_directories(${LLVM_INCLUDE_DIRS})
#
# # for files generated from tablegen
# include_directories(${CMAKE_BINARY_DIR}/mlir)
#
# # add dialect and passes add_subdirectory(mlir)
#
# # add compiler add_executable(compiler compiler_test.cpp)
# llvm_update_compile_flags(compiler) # add no-rtti when needed
# target_link_libraries( compiler PRIVATE MLIRGGML MLIRParser MLIRSupport MLIRIR
# MLIRFuncDialect MLIRFunctionInterfaces MLIRArithDialect MLIRPass)

add_executable(test_llama test_llama.cpp)
target_link_libraries(test_llama llama)
