local preset = "debug"
-- local preset = "release"
local model = "../gguf/didedoshka.gguf"
-- local model = "../../stories15M.gguf"

return {
    compiler = {
        dir = "build_" .. preset,
        build = "cmake --build . --target compiler",
        run = "./source/compiler " .. model .. " -debug -output=tensor_tosa",
    },

    cmake = {
        build = "rm -rf build_" .. preset,
        run = "cmake --preset " .. preset,
    },

    mlir_tensor_to_affine = {
        run = "mlir-opt main.mlir --convert-linalg-to-affine-loops",
    },

    mlir_check = {
        dir = "data",
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
