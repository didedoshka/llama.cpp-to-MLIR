#include "Utils.h"

#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"

constexpr int TENSOR_OUTPUT_WIDTH = 3;

std::string ggmlTensorFormat(const ggml_tensor *tensor) {
    const int64_t *ne = tensor->ne;
    std::string str;
    llvm::raw_string_ostream strBuilder(str);
    strBuilder << std::string(tensor->name) << ' ' << ggml_get_type_traits(tensor->type)->type_name << '\n';
    strBuilder << ne[3] << ' ' << ne[2] << ' ' << ne[1] << ' ' << ne[0] << '\n';
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        strBuilder << "[\n";
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == TENSOR_OUTPUT_WIDTH && ne[2] > 2 * TENSOR_OUTPUT_WIDTH) {
                strBuilder << " ..., \n";
                i2 = ne[2] - TENSOR_OUTPUT_WIDTH;
            }
            strBuilder << " [\n";
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == TENSOR_OUTPUT_WIDTH && ne[1] > 2 * TENSOR_OUTPUT_WIDTH) {
                    strBuilder << "  ..., \n";
                    i1 = ne[1] - TENSOR_OUTPUT_WIDTH;
                }
                strBuilder << "  [";
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == TENSOR_OUTPUT_WIDTH && ne[0] > 2 * TENSOR_OUTPUT_WIDTH) {
                        strBuilder << "..., ";
                        i0 = ne[0] - TENSOR_OUTPUT_WIDTH;
                    }
                    const float v = ggmlTensorGet<float>(tensor, i0, i1, i2, i3);
                    strBuilder << llvm::formatv("{0:F3}", v);
                    if (i0 < ne[0] - 1)
                        strBuilder << ", ";
                }
                strBuilder << "],\n";
            }
            strBuilder << " ],\n";
        }
        strBuilder << "]\n";
    }
    return str;
}

std::string compareGGMLAndMLIRResults(const ggml_tensor *ggmlResult,
                                      mlir::OwningMemRef<float, nDims> &mlirResult) {
    const int64_t *ggmlShape = ggmlResult->ne;
    const int64_t *mlirShape = mlirResult->sizes;

    std::string difference;
    if (std::tuple{ggmlShape[3], ggmlShape[2], ggmlShape[1], ggmlShape[0]} !=
        std::tuple{mlirShape[0], mlirShape[1], mlirShape[2], mlirShape[3]}) {
        return llvm::formatv("Shapes aren't the same: ggml shape = {}x{}x{}x{} mlir shape = {}x{}x{}x{}",
                             ggmlShape[3], ggmlShape[2], ggmlShape[1], ggmlShape[0], mlirShape[0],
                             mlirShape[1], mlirShape[2], mlirShape[3]);
    }

    for (int64_t i3 = 0; i3 < ggmlShape[3]; i3++) {
        for (int64_t i2 = 0; i2 < ggmlShape[2]; i2++) {
            for (int64_t i1 = 0; i1 < ggmlShape[1]; i1++) {
                for (int64_t i0 = 0; i0 < ggmlShape[0]; i0++) {
                    const float ggmlV = ggmlTensorGet<float>(ggmlResult, i0, i1, i2, i3);
                    const float mlirV = mlirResult[{i3, i2, i1, i0}];
                    if (ggmlV != mlirV) {
                        return llvm::formatv("Values at position [{}, {}, {}, {}] aren't the same. ggml "
                                             "value: {}, mlir value {}",
                                             i3, i2, i1, i0, ggmlV, mlirV);
                    }
                }
            }
        }
    }
    return "";
}

std::string mlirTensorFormat(mlir::OwningMemRef<float, nDims> &mlirResult) {
    const int64_t *ne = mlirResult->sizes;
    std::string str;
    llvm::raw_string_ostream strBuilder(str);
    // strBuilder << std::string(tensor->name) << ' ' << ggml_get_type_traits(tensor->type)->type_name <<
    // '\n';
    strBuilder << ne[0] << ' ' << ne[1] << ' ' << ne[2] << ' ' << ne[3] << '\n';
    for (int64_t i3 = 0; i3 < ne[0]; i3++) {
        strBuilder << "[\n";
        for (int64_t i2 = 0; i2 < ne[1]; i2++) {
            if (i2 == TENSOR_OUTPUT_WIDTH && ne[1] > 2 * TENSOR_OUTPUT_WIDTH) {
                strBuilder << " ..., \n";
                i2 = ne[1] - TENSOR_OUTPUT_WIDTH;
            }
            strBuilder << " [\n";
            for (int64_t i1 = 0; i1 < ne[2]; i1++) {
                if (i1 == TENSOR_OUTPUT_WIDTH && ne[2] > 2 * TENSOR_OUTPUT_WIDTH) {
                    strBuilder << "  ..., \n";
                    i1 = ne[2] - TENSOR_OUTPUT_WIDTH;
                }
                strBuilder << "  [";
                for (int64_t i0 = 0; i0 < ne[3]; i0++) {
                    if (i0 == TENSOR_OUTPUT_WIDTH && ne[3] > 2 * TENSOR_OUTPUT_WIDTH) {
                        strBuilder << "..., ";
                        i0 = ne[3] - TENSOR_OUTPUT_WIDTH;
                    }
                    const float v = mlirResult[{i3, i2, i1, i0}];
                    strBuilder << llvm::formatv("{0:F3}", v);
                    if (i0 < ne[3] - 1)
                        strBuilder << ", ";
                }
                strBuilder << "],\n";
            }
            strBuilder << " ],\n";
        }
        strBuilder << "]\n";
    }
    return str;
}
