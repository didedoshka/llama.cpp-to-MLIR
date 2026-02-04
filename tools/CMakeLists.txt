add_executable(ggml-mlir main.cpp)

target_link_libraries(ggml-mlir PRIVATE ggml::ggml ggml::ggml-base llama)