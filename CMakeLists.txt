cmake_minimum_required(VERSION 3.21)

project(ggml-mlir C CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_WARN_UNUSED_CLI YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(llama)

add_executable(ggml-mlir source/main.cpp)
target_link_libraries(ggml-mlir PRIVATE llama)

add_executable(ggml_simple source/ggml_simple.cpp)
target_link_libraries(ggml_simple PRIVATE llama)
