-- local preset = "debug"
local preset = "sanitized"
-- local preset = "release"
--
-- local model = "../gguf/didedoshka.gguf"
local model = "../../stories15M.gguf"
--
-- local output = "ggml"
-- local output = "tensor_tosa"
-- local output = "llvmmlir"
local output = "debug"

return {
    compiler = {
        dir = "build_" .. preset,
        build = "cmake --build . --target compiler",
        run = "ASAN_OPTIONS=detect_container_overflow=0 ./source/compiler " .. model ..
            " -output-kind=" .. output ..
            " -debug" ..
            " -debug-only=ggml_mlir,builtinattributes" ..
            -- " -input-mlir=../data/main.mlir" ..
            " -output-mlir=../data/2.mlir" ..
            " -execute-n-operations=1" ..
            "",
    },

    cmake = {
        build = "rm -rf build_" .. preset,
        run = "cmake --preset " .. preset,
    },

    mlir_bufferize = {
        dir = "data",
        -- run = [[mlir-opt main.mlir -one-shot-bufferize="bufferize-function-boundaries"]]
        run = [[mlir-opt main.mlir -fold-tensor-subset-ops]]
    },

    mlir_check = {
        dir = "data",
        run = "mlir-opt main.mlir -allow-unregistered-dialect",
    },

    -- https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir
    mlir_build = {
        dir = "data",
        run = [[
        mlir-opt main.mlir -o compiled.mlir
        -one-shot-bufferize="bufferize-function-boundaries"
        -buffer-deallocation-pipeline -convert-bufferization-to-memref
        -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata
        -lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts
        ]],
    },

    mlir_run = {
        dir = "data",
        run =
        "mlir-cpu-runner compiled.mlir -e compute_graph_forward -entry-point-result=void -shared-libs=$MLIR_C_RUNNER_UTILS,$MLIR_RUNNER_UTILS",
    },

    mlir_build_and_run = {
        dir = "data",
        build = [[
        mlir-opt main.mlir -o compiled.mlir
        -one-shot-bufferize="bufferize-function-boundaries"
        -buffer-deallocation-pipeline -convert-bufferization-to-memref
        -expand-strided-metadata
        -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf 
        -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts
        ]],
        run =
        "mlir-cpu-runner compiled.mlir -e compute_graph_forward -entry-point-result=void -shared-libs=$MLIR_C_RUNNER_UTILS,$MLIR_RUNNER_UTILS",
    }
}
