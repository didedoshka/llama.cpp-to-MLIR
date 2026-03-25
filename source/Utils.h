#pragma once

#include "ggml.h"
#include <string>

#include "mlir/ExecutionEngine/MemRefUtils.h"

float ggmlTensorGet(const ggml_tensor *tensor, size_t i0, size_t i1, size_t i2, size_t i3);
std::string ggmlTensorFormat(const ggml_tensor *tensor);

std::string compareGGMLAndMLIRResults(const ggml_tensor *ggmlResult,
                                      mlir::OwningMemRef<float, 4> &mlirResult);

std::string mlirTensorFormat(mlir::OwningMemRef<float, 4> &mlirResult);
