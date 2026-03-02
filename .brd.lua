return {
    main_15m = {
        dir = "build",
        build = "cmake --build . --target ggml-mlir",
        run = "./ggml-mlir ../../stories15M.gguf",
    },

    main_7b = {
        dir = "build",
        build = "cmake --build . --target ggml-mlir",
        run = "./ggml-mlir ../../ggml-model-q2_k.gguf",
    },

    cmake = {
        dir = vim.fn.getcwd,
        run = "cmake -B build -DFETCHCONTENT_SOURCE_DIR_LLAMA=../llama.cpp -DGGML_METAL=OFF",
    },

    ggml_simple = {
        dir = "build",
        build = "cmake --build . --target ggml_simple",
        run = "./ggml_simple",
    },

    mlir_tensor_to_affine = {
        run = "mlir-opt main.mlir --convert-linalg-to-affine-loops",
        -- run = "mlir-opt main.mlir --one-shot-bufferize --convert-linalg-to-affine-loops",
    },

    mlir_check = {
        -- run = "mlir-opt main.mlir",
        run = "mlir-opt main.mlir -allow-unregistered-dialect",
    },

    mlir_compile_and_run = {
        build = "mlir-opt main.mlir --convert-linalg-to-affine-loops --convert-to-llvm -o compiled.mlir",
        run = "mlir-cpu-runner -e main -entry-point-result=i32 compiled.mlir"
    }
}
