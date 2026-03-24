#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
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

namespace {
enum InputType { GGUF,
                 MLIR };
} // namespace

namespace cl = llvm::cl;

cl::OptionCategory
    CompilerCategory("Compiler Options", "Options for controlling the compilation process.");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"),
                                          cl::cat(CompilerCategory));

namespace {
enum Output { GGML,
              TensorTOSA };
} // namespace
static cl::opt<enum Output> outputType(
    "output", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(GGML, "ggml", "output ggml dialect")),
    cl::values(clEnumValN(TensorTOSA, "tensor_tosa", "output lowered to tensor_tosa")));

static llvm::LogicalResult RunPasses(mlir::ModuleOp module) {
    if (outputType == Output::TensorTOSA) {
        mlir::PassManager pm(module->getName());
        pm.addPass(mlir::ggml::createLoweringPass());
        return pm.run(module);
    } else {
        return llvm::success();
    }
}

int main(int argc, char **argv) {
    // cl::HideUnrelatedOptions(CompilerCategory);
    cl::ParseCommandLineOptions(argc, argv);
    std::cout << inputFilename << '\n';

    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::ggml::GGMLDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);

    // module
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

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

    if (llvm::failed(RunPasses(module))) {
        llvm::errs() << "RunPasses failed";
    }

    if (llvm::failed(mlir::verify(module))) {
        llvm::errs() << "module verification failed";
    }
    module->dump();

    return 0;
}
