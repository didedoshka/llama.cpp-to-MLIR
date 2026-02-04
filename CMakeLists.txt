cmake_minimum_required(VERSION 3.14)
project(ggml-mlir C CXX)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ggml CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

include_directories("${GGML_INCLUDE_DIR}")
include_directories("${LLAMA_INCLUDE_DIR}")

add_subdirectory(tools)
