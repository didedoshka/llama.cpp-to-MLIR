#include "MLIRGen.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DebugLog.h"

#include "Utils.h"
#include "ggml-cpu.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "MLIRGGML/GGMLOps.h"

MLIRGen::MLIRGen(mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ModuleOp &module) : context(context), builder(builder), module(module) {
    func = mlir::func::FuncOp::create(
        builder,
        builder.getUnknownLoc(),
        "compute_graph_forward",
        builder.getFunctionType({}, {}));
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
}

mlir::RankedTensorType MLIRGen::getGGMLTensorType(const ggml_tensor *t) {
    // TODO: other types
    return mlir::RankedTensorType::get({t->ne[3], t->ne[2], t->ne[1], t->ne[0]}, builder.getF32Type());
}

static int64_t getGGMLTensorNumberOfElements(const ggml_tensor *t) {
    int64_t result = 1;
    for (int i = 0; i < 4; ++i) {
        result *= t->ne[i];
    }
    return result;
}

// TODO: optimize
// TODO: other types
static std::vector<float> getGGMLTensorData(const ggml_tensor *t) {
    std::vector<float> result(getGGMLTensorNumberOfElements(t));
    const int64_t *ne = t->ne;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = ggmlTensorGet(t, i0, i1, i2, i3);
                    result[ne[3] * i3 + ne[2] * i2 + ne[1] * i1 + i0] = v;
                }
            }
        }
    }
    return result;
}

void MLIRGen::addOp(const ggml_tensor *t) {
    llvm::SmallVector<mlir::Value, 2> sources;

    for (int i = 0; t->src[i] != nullptr; ++i) {
        auto source = t->src[i];
        auto it = tensors.find(source->name);
        if (it != tensors.end()) {
            sources.push_back(it->getValue());
        } else {
            auto rankedTensorType = getGGMLTensorType(t);
            auto dataRaw = getGGMLTensorData(source);
            auto data = mlir::DenseElementsAttr::get(rankedTensorType, llvm::ArrayRef(dataRaw));
            auto op = mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), data);
            tensors.insert({source->name, op});
            sources.push_back(op);
            last = op;
        }
    }

    // switch (t->op) {
    // default:;
    // }

    auto result = mlir::ggml::AddOp::create(builder, builder.getUnknownLoc(), sources[0], sources[1]);
    last = result;
}

void MLIRGen::finish() {
    func.setFunctionType(builder.getFunctionType({}, {last.getType()}));
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), {last});
}
