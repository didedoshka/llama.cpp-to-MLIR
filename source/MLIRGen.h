#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <ggml.h>

class MLIRGen {
  public:
    MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ModuleOp &module);
    void addOp(const ggml_tensor *t);
    void finish();

  private:
    mlir::RankedTensorType getGGMLTensorType(const ggml_tensor *t);
    mlir::MLIRContext &context;
    mlir::OpBuilder &builder;
    mlir::ModuleOp &module;

    mlir::Value last;
    mlir::func::FuncOp func;

    llvm::StringMap<mlir::Value> tensors;
};
