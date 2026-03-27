#pragma once

#include "ggml.h"
#include <string>

#include "mlir/ExecutionEngine/MemRefUtils.h"

#include "Defines.h"

template <typename cppType>
cppType ggmlTensorGet(const ggml_tensor *tensor, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0];
    uint8_t *data = static_cast<uint8_t *>(tensor->data);
    return *(const cppType *)&data[i];
}
std::string ggmlTensorFormat(const ggml_tensor *tensor);

std::string compareGGMLAndMLIRResults(const ggml_tensor *ggmlResult,
                                      mlir::OwningMemRef<float, nDims> &mlirResult);

std::string mlirTensorFormat(mlir::OwningMemRef<float, nDims> &mlirResult);
