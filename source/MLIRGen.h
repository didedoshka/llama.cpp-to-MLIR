#include "llvm/ADT/StringMap.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include <ggml.h>

class MLIRGen {
  public:
    MLIRGen(mlir::OpBuilder &builder);
    void appendOp(const ggml_tensor *t);
    void finish();

  private:
    mlir::OpBuilder &builder;

    mlir::Value last;
    mlir::func::FuncOp func;

    llvm::StringMap<mlir::Value> tensors;
};


class MLIRGen2 {
  public:
    MLIRGen2(mlir::OpBuilder &builder);
    void appendOp(const ggml_tensor *t);
    void finish();

  private:
    mlir::OpBuilder &builder;
    //
    // mlir::Value last;
    // mlir::func::FuncOp func;
    //
    // llvm::StringMap<mlir::Value> tensors;
};
