#include "GGMLOps.h"
#include "GGMLDialect.h"

#define GET_OP_CLASSES
#include "GGMLOps.cpp.inc"

using namespace mlir;
using namespace mlir::ggml;

void GetRowsOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value values,
                      mlir::Value indices) {
    auto valuesType = llvm::dyn_cast<mlir::RankedTensorType>(values.getType());
    auto valuesShape = valuesType.getShape();

    auto indicesType = llvm::dyn_cast<mlir::RankedTensorType>(indices.getType());
    auto indicesShape = indicesType.getShape();

    auto resultType =
        RankedTensorType::get({1, 1, indicesShape[3], valuesShape[3]}, valuesType.getElementType());
    GetRowsOp::build(builder, state, resultType, values, indices);
}

void MulMatOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value A,
                      mlir::Value B) {
    auto AType = llvm::dyn_cast<mlir::RankedTensorType>(A.getType());
    auto AShape = AType.getShape();

    auto BType = llvm::dyn_cast<mlir::RankedTensorType>(B.getType());
    auto BShape = BType.getShape();

    auto resultType =
        RankedTensorType::get({1, 1, BShape[2], AShape[2]}, AType.getElementType());
    GetRowsOp::build(builder, state, resultType, A, B);
}
