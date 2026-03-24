return {
    test_gguf = {
        dir = "",
        build = "python gguf/writer.py gguf/didedoshka.gguf && cmake --build build_debug --target test_llama",
        run = "./build_debug/test_llama gguf/didedoshka.gguf",
    },

    test_stories = {
        dir = "build_debug",
        build = "cmake --build . --target test_llama",
        run = "./test_llama ../../stories15M.gguf",
    },

    test_llama = {
        dir = "build_debug",
        build = "cmake --build . --target test_llama",
        run = "./test_llama ../gguf/didedoshka.gguf",
    },

    compiler = {
        dir = "build_debug",
        build = "cmake --build . --target compiler",
        run = "./compiler -debug -output=tensor_tosa",
    },

    compiler_release = {
        dir = "build_release",
        build = "cmake --build . --target compiler",
        run = "./compiler",
    },

    main_15m = {
        dir = "build",
        build = "cmake --build . --target compiler",
        run = "./compiler ../../stories15M.gguf",
    },

    main_7b = {
        dir = "build",
        build = "cmake --build . --target ggml-mlir",
        run = "./ggml-mlir ../../ggml-model-q2_k.gguf",
    },

    cmake_cl = {
        run = "cmake -B build -DFETCHCONTENT_SOURCE_DIR_LLAMA=../llama.cpp -DGGML_METAL=OFF",
    },

    cmake_debug = {
        build = "rm -rf build_debug",
        run = "cmake --preset debug",
    },

    cmake_release = {
        build = "rm -rf build_release",
        run = "cmake --preset release",
    },

    mlir_cat = {
        dir = "build",
        build = "cmake --build . --target mlir-cat2",
        run = "./mlir-cat2",
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
        dir = "data",
        -- run = "mlir-opt main.mlir",
        run = "mlir-opt tensor.mlir -allow-unregistered-dialect",
    },

    -- https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir
    mlir_build = {
        dir = "data",
        run = [[
        mlir-opt add.mlir -o compiled.mlir
        -one-shot-bufferize="bufferize-function-boundaries"
        -buffer-deallocation-pipeline -convert-bufferization-to-memref
        -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata
        -lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts
        ]],
    },

    mlir_run = {
        dir = "data",
        run =
        "mlir-cpu-runner compiled.mlir -e main -entry-point-result=void -shared-libs=$MLIR_C_RUNNER_UTILS,$MLIR_RUNNER_UTILS",
    },

    mlir_compile_and_run = {
        build = "mlir-opt main.mlir --convert-linalg-to-affine-loops --convert-to-llvm -o compiled.mlir",
        run = "mlir-cpu-runner -e main -entry-point-result=i32 compiled.mlir"
    }
}
