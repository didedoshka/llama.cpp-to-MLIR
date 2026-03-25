#include <cstdlib>
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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include "MLIRGGML/GGMLDialect.h"
#include "MLIRGGML/GGMLOps.h"
#include "MLIRGGML/Passes.h"

#include "MLIRGen.h"
#include "Utils.h"

#include "Defines.h"

namespace {
enum InputType { GGUF, MLIR };
} // namespace

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                          cl::value_desc("filename"));

namespace {
enum Output { GGML, TensorTOSA, LLVMMLIR, Debug };
} // namespace
static cl::opt<enum Output>
    outputType("output", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(GGML, "ggml", "output ggml dialect")),
               cl::values(clEnumValN(TensorTOSA, "tensor_tosa", "output lowered to tensor_tosa")),
               cl::values(clEnumValN(LLVMMLIR, "llvmmlir", "output lowered to llvmmlir")),
               cl::values(clEnumValN(Debug, "debug", "compile and compare result with llama.cpp")));

static llvm::LogicalResult RunPasses(mlir::ModuleOp &module) {
    if (outputType == Output::GGML) {
        return llvm::success();
    }
    mlir::PassManager pm(module->getName());
    if (outputType >= Output::TensorTOSA) {
        pm.addPass(mlir::ggml::createLoweringPass());
    }
    if (outputType >= Output::LLVMMLIR) {
        // -one-shot-bufferize="bufferize-function-boundaries"
        mlir::bufferization::OneShotBufferizePassOptions oneShotBufferizePassOptions;
        oneShotBufferizePassOptions.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(oneShotBufferizePassOptions));
        // -buffer-deallocation-pipeline
        mlir::bufferization::BufferDeallocationPipelineOptions bufferDeallocationPipelineOptions;
        mlir::bufferization::buildBufferDeallocationPipeline(pm, bufferDeallocationPipelineOptions);
        // -convert-bufferization-to-memref
        pm.addPass(mlir::createConvertBufferizationToMemRefPass());
        // -convert-linalg-to-loops
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        // -convert-scf-to-cf
        pm.addPass(mlir::createSCFToControlFlowPass());
        // -expand-strided-metadata
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        // -lower-affine
        pm.addPass(mlir::createLowerAffinePass());
        // -convert-arith-to-llvm
        pm.addPass(mlir::createArithToLLVMConversionPass());
        // -finalize-memref-to-llvm
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        // -convert-func-to-llvm
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        // -convert-cf-to-llvm
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        // -reconcile-unrealized-casts
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }
    return pm.run(module);
}

static llvm::LogicalResult runJIT(mlir::ModuleOp &module, mlir::OwningMemRef<float, 4> &res) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    // engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    auto resPointer = &*res;
    auto invocationResult =
        engine->invoke("compute_graph_forward", mlir::ExecutionEngine::result(resPointer));
    if (invocationResult) {
        return llvm::failure();
    }

    return llvm::success();
}

struct DebugData {
    int nOperations = 0;
    ggml_tensor *result;

    ggml_tensor resultCopy;
    void *resultData;
};

struct UserData {
    MLIRGen mlirGen;
    DebugData debugData;
};

void llamaRun(UserData &userData, ggml_backend_sched_eval_callback cb) {
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
// 01. ADD linalg.add
// 02. CPY     a -> b, return view(b) what is view in ggml? looks like copy-on-write
// 03. FLASH_ATTN_EXT
// 04. GET_ROWS https://mlir.llvm.org/docs/Dialects/TOSA/
// https://www.rdocumentation.org/packages/ggmlR/versions/0.6.1/topics/ggml_get_rows
// 05. GLU ggml.h:1253
// 06. MUL linalg.mul
// 07. MUL_MAT linalg.matmul
// 08. PERMUTE multidimensional transpose. in ggml is done without copying
// sometimes (and no-op on most backends), can it create problems?
// 09. RESHAPE tensor.reshape
// 10. RMS_NORM
// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
// 11. ROPE ggml.h:1747
// 12. SET_ROWS
// 13. VIEW ggml.h:830?

// constexpr int64_t executeNOperations = INT32_MAX;
constexpr int64_t executeNOperations = 1;

int main(int argc, char **argv) {
    // cl::HideUnrelatedOptions(CompilerCategory);
    cl::ParseCommandLineOptions(argc, argv);

    mlir::MLIRContext context;
    // TODO: figure out the difference beetween register and getOrLoad, remove registerAllDialects
    mlir::registerAllDialects(context);
    // mlir::DialectRegistry dialectRegistry;
    // dialectRegistry.insert<mlir::ggml::GGMLDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect>();
    // context.appendDialectRegistry(dialectRegistry);
    context.getOrLoadDialect<mlir::ggml::GGMLDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);

    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

    UserData userData{MLIRGen(context, builder, module), DebugData()};
    MLIRGen &mlirGen = userData.mlirGen;
    DebugData &debugData = userData.debugData;

    llamaRun(userData, [](ggml_tensor *t, bool ask, void *rawUserData) {
        UserData *userData = static_cast<UserData *>(rawUserData);
        MLIRGen &mlirGen = userData->mlirGen;
        DebugData &debugData = userData->debugData;
        if (debugData.nOperations == 2 * executeNOperations) {
            LDBG() << debugData.nOperations << " " << ask;
            if (ask) {
                size_t nbytes = ggml_nbytes(debugData.result);
                // it memory leaks, no problem with that
                memcpy(&debugData.resultCopy, debugData.result, sizeof(*debugData.result));

                debugData.resultData = malloc(nbytes);
                memcpy(debugData.resultData, debugData.result->data, nbytes);
                debugData.resultCopy.data = debugData.resultData;

                debugData.result = &debugData.resultCopy;
                return true;
            } else {
                return false;
            }
        }
        ++debugData.nOperations;

        if (ask) {
            LDBG() << "Adding operation " << ggml_op_name(t->op) << '\n';

            const ggml_tensor *src0 = t->src[0];
            const ggml_tensor *src1 = t->src[1];

            LDBG() << ggmlTensorFormat(src0);
            if (src1 != nullptr) {
                LDBG() << ggmlTensorFormat(src1);
            }
            mlirGen.addOp(t);
            return true;
        }
        LDBG() << ggmlTensorFormat(t);
        debugData.result = t;

        return true;
    });
    mlirGen.finish();

    if (llvm::failed(RunPasses(module))) {
        llvm::errs() << "RunPasses failed";
    }

    if (llvm::failed(mlir::verify(module))) {
        llvm::errs() << "module verification failed";
    }

    module->dump();

    LDBG() << debugData.result;

    if (outputType == Output::Debug) {
        mlir::OwningMemRef<float, 4> mlirResult(llvm::ArrayRef<int64_t>{0, 0, 0, 0});
        if (llvm::failed(runJIT(module, mlirResult))) {
            llvm::errs() << "JIT failed";
        }
        LDBG() << "JIT succeeded\n";
        LDBG() << "GGML result tensor\n" << ggmlTensorFormat(debugData.result);
        LDBG() << "MLIR result tensor\n" << mlirTensorFormat(mlirResult);

        auto difference = compareGGMLAndMLIRResults(debugData.result, mlirResult);
        if (difference != "") {
            LDBG() << "Failure. MLIR and GGML produced different results";
            LDBG() << difference;
        } else {
            LDBG() << "Success. MLIR and GGML produced the same result";
        }
    }

    return 0;
}
