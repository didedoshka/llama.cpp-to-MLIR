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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include "MLIRGGML/GGMLDialect.h"
#include "MLIRGGML/GGMLOps.h"
#include "MLIRGGML/Passes.h"

#include "MLIRGen.h"
#include "Utils.h"

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

llvm::LogicalResult RunPasses(mlir::ModuleOp &module) {
    if (outputType == Output::TensorTOSA) {
        mlir::PassManager pm(module->getName());
        pm.addPass(mlir::ggml::createLoweringPass());
        return pm.run(module);
    } else {
        return llvm::success();
    }
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
                    const float v = ggmlTensorGet(tensor, i0, i1, i2, i3);
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

struct DebugData {
    int nOperations = 0;
    ggml_tensor *result;
};

struct UserData {
    MLIRGen mlirGen;
    DebugData debugData;
};

void llamaRun(UserData userData, ggml_backend_sched_eval_callback cb) {
    llama_model_params model_params = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(inputFilename.data(), model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval_user_data = &userData;
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

// constexpr int64_t executeNOperations = INT32_MAX;
constexpr int64_t executeNOperations = 1;

int main(int argc, char **argv) {
    // cl::HideUnrelatedOptions(CompilerCategory);
    cl::ParseCommandLineOptions(argc, argv);

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::ggml::GGMLDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);

    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

    UserData userData{MLIRGen(context, builder, module), DebugData()};
    MLIRGen& mlirGen = userData.mlirGen;

    llamaRun(userData, [](ggml_tensor *t, bool ask, void *rawUserData) {
        UserData *userData = static_cast<UserData *>(rawUserData);
        MLIRGen &mlirGen = userData->mlirGen;
        DebugData &debugData = userData->debugData;
        if (debugData.nOperations == 2 * executeNOperations) {
            return false;
        }
        ++debugData.nOperations;

        if (ask) {
            LDBG() << "Adding operation " << ggml_op_name(t->op) << '\n';

            const ggml_tensor *src0 = t->src[0];
            const ggml_tensor *src1 = t->src[1];

            LDBG() << formatGGMLTensor(src0);
            if (src1 != nullptr) {
                LDBG() << formatGGMLTensor(src1);
            }
            mlirGen.addOp(t);
            return true;
        }
        LDBG() << formatGGMLTensor(t);
        // debugData.result = t;

        return true;
    });
    mlirGen.finish();

    //
    // if (llvm::failed(mlirGen.RunPasses())) {
    //     llvm::errs() << "RunPasses failed";
    // }
    //
    if (llvm::failed(mlir::verify(module))) {
        llvm::errs() << "module verification failed";
    }

    module->dump();

    return 0;
}
