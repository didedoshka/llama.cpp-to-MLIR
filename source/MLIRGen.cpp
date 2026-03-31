#include "MLIRGen.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DebugLog.h"

#include "Utils.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "MLIRGGML/GGMLOps.h"

#include "Defines.h"

MLIRGen::MLIRGen(mlir::OpBuilder &builder) : builder(builder) {
    func = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(), "compute_graph_forward",
                                      builder.getFunctionType({}, {}));
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
}

mlir::RankedTensorType getGGMLTensorType(const ggml_tensor *t, mlir::Type mlirType) {
    return mlir::RankedTensorType::get({t->ne[3], t->ne[2], t->ne[1], t->ne[0]}, mlirType);
}

static int64_t getGGMLTensorNumberOfElements(const ggml_tensor *t) {
    int64_t result = 1;
    for (int i = 0; i < 4; ++i) {
        result *= t->ne[i];
    }
    return result;
}

// TODO: optimize
template <typename cppType> static std::vector<cppType> getGGMLTensorData(const ggml_tensor *t) {
    auto numberOfElements = getGGMLTensorNumberOfElements(t);
    std::vector<cppType> result(numberOfElements);
    const int64_t *ne = t->ne;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const cppType v = ggmlTensorGet<cppType>(t, i0, i1, i2, i3);
                    result[i3 * ne[2] * ne[1] * ne[0] + i2 * ne[1] * ne[0] + i1 * ne[0] + i0] = v;
                }
            }
        }
    }
    return result;
}

template <ggml_type ggmlType, typename cppType>
mlir::DenseElementsAttr getDenseElementsAttr(ggml_tensor *t, mlir::Type mlirType) {
    auto rankedTensorType = getGGMLTensorType(t, mlirType);
    auto dataRaw = getGGMLTensorData<cppType>(t);
    return mlir::DenseElementsAttr::get(rankedTensorType, llvm::ArrayRef(dataRaw));
}

void MLIRGen::appendOp(const ggml_tensor *t) {
    llvm::SmallVector<mlir::Value, 2> sources;

    for (int i = 0; t->src[i] != nullptr; ++i) {
        auto source = t->src[i];
        auto it = tensors.find(source->name);
        if (it != tensors.end()) {
            sources.push_back(it->getValue());
        } else {
            mlir::DenseElementsAttr data;
            switch (source->type) {
            case GGML_TYPE_F32:
                data = getDenseElementsAttr<GGML_TYPE_F32, float>(source, builder.getF32Type());
                break;
            case GGML_TYPE_I32:
                data = getDenseElementsAttr<GGML_TYPE_I32, int32_t>(source, builder.getI32Type());
                break;
            default:;
            }
            last = mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), data);
            tensors.insert({source->name, last});
            sources.push_back(last);
        }
    }

    switch (t->op) {
    case GGML_OP_ADD:
        last = mlir::ggml::AddOp::create(builder, builder.getUnknownLoc(), sources[0], sources[1]);
        break;
    case GGML_OP_GET_ROWS:
        last = mlir::ggml::GetRowsOp::create(builder, builder.getUnknownLoc(), sources[0], sources[1]);
        break;
    case GGML_OP_MUL_MAT:
        last = mlir::ggml::MulMatOp::create(builder, builder.getUnknownLoc(), sources[0], sources[1]);
        break;
    default:
        LDBG() << "Operation " << ggml_op_name(t->op) << " is not supported";
        return;
    }
    tensors.insert_or_assign(t->name, last);
}

void MLIRGen::finish() {
    func.setFunctionType(builder.getFunctionType({}, {last.getType()}));
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), {last});
}
