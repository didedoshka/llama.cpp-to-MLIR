#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/GGMLDialect.h"
#include "mlir/GGMLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Passes.h"

#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

namespace {
enum InputType { GGUF,
                 MLIR };
} // namespace

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"));

namespace {
enum Output { GGML,
              TensorTOSA,
              Debug };
} // namespace
static cl::opt<enum Output> outputType(
    "output", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(GGML, "ggml", "output ggml dialect")),
    cl::values(clEnumValN(TensorTOSA, "tensor_tosa", "output lowered to tensor_tosa")),
    cl::values(clEnumValN(Debug, "debugggg", "compile and compare result with llama.cpp")));

struct MLIRGen {
    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;

    MLIRGen() : builder(&context) {
        context.getOrLoadDialect<mlir::ggml::GGMLDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        // module
        module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(module.getBody());
    }

    void addOp() {
        auto rankedTensorType = mlir::RankedTensorType::get({2, 2}, builder.getF32Type());

        // function
        auto func = mlir::func::FuncOp::create(
            builder,
            builder.getUnknownLoc(),
            "compute_graph_forward",
            builder.getFunctionType({}, rankedTensorType));
        mlir::Block *entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        llvm::SmallVector<float, 4> dataRaw = {1, 2, 3, 4};
        auto data = mlir::DenseElementsAttr::get(rankedTensorType, llvm::ArrayRef(dataRaw));
        // firstTensor
        auto firstTensor = mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), data);
        // firstTensor
        auto secondTensor = mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), data);
        // AddOp
        auto result = mlir::ggml::AddOp::create(builder, builder.getUnknownLoc(), firstTensor, secondTensor);
        // ReturnOp
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), {result});
    }

    llvm::LogicalResult RunPasses() {
        if (outputType == Output::TensorTOSA) {
            mlir::PassManager pm(module->getName());
            pm.addPass(mlir::ggml::createLoweringPass());
            return pm.run(module);
        } else {
            return llvm::success();
        }
    }
};

static float ggml_get_float_value(const ggml_tensor *tensor, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0];
    float v;
    uint8_t *data = static_cast<uint8_t *>(tensor->data);
    if (tensor->type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *)&data[i]);
    } else if (tensor->type == GGML_TYPE_F32) {
        v = *(const float *)&data[i];
    } else if (tensor->type == GGML_TYPE_I64) {
        v = (float)*(const int64_t *)&data[i];
    } else if (tensor->type == GGML_TYPE_I32) {
        v = (float)*(const int32_t *)&data[i];
    } else if (tensor->type == GGML_TYPE_I16) {
        v = (float)*(const int16_t *)&data[i];
    } else if (tensor->type == GGML_TYPE_I8) {
        v = (float)*(const int8_t *)&data[i];
    } else {
        LDBG() << "unexpected type: " << ggml_get_type_traits(tensor->type)->type_name;
    }
    return v;
}

constexpr int TENSOR_OUTPUT_WIDTH = 3;

static std::string formatGGMLTensor(const ggml_tensor *tensor) {
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
                    const float v = ggml_get_float_value(tensor, i0, i1, i2, i3);
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

void llamaRun(MLIRGen &mlirGen, ggml_backend_sched_eval_callback cb) {
    llama_model_params model_params = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(inputFilename.data(), model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval_user_data = &mlirGen;
    ctx_params.cb_eval = cb;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    std::vector<llama_token> tokens = {1, 2, 3};
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
}

// https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md
// 13 different operations
// ADD linalg.add
// CPY     // a -> b, return view(b) what is view in ggml? looks like copy-on-write
// FLASH_ATTN_EXT
// GET_ROWS https://mlir.llvm.org/docs/Dialects/TOSA/ https://www.rdocumentation.org/packages/ggmlR/versions/0.6.1/topics/ggml_get_rows
// GLU ggml.h:1253
// MUL linalg.mul
// MUL_MAT linalg.matmul
// PERMUTE multidimensional transpose. in ggml is done without copying sometimes (and no-op on most backends), can it create problems?
// RESHAPE tensor.reshape
// RMS_NORM https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
// ROPE ggml.h:1747
// SET_ROWS
// VIEW ggml.h:830?

int main(int argc, char **argv) {
    // cl::HideUnrelatedOptions(CompilerCategory);
    cl::ParseCommandLineOptions(argc, argv);

    MLIRGen mlirGen{};

    llamaRun(mlirGen, [](ggml_tensor *t, bool ask, void *userData) {
        if (ask) {
            return true;
        }
        MLIRGen *mlirGen = static_cast<MLIRGen *>(userData);
        LDBG() << ggml_op_name(t->op) << '\n';

        const ggml_tensor *src0 = t->src[0];
        const ggml_tensor *src1 = t->src[1];

        LDBG() << formatGGMLTensor(src0);
        if (src1 != nullptr) {
            LDBG() << formatGGMLTensor(src1);
        }
        LDBG() << formatGGMLTensor(t);

        return true;
    });

    // mlirGen.addOp();
    //
    // if (llvm::failed(mlirGen.RunPasses())) {
    //     llvm::errs() << "RunPasses failed";
    // }
    //
    // if (llvm::failed(mlir::verify(mlirGen.module))) {
    //     llvm::errs() << "module verification failed";
    // }
    //
    // mlirGen.module->dump();

    return 0;
}
