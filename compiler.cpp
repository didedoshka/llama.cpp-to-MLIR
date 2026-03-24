#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <string>

#include "llvm/Support/CommandLine.h"

#include "mlir/IR/MLIRContext.h"

namespace {
enum InputType { GGUF,
                 MLIR };
}  // namespace

namespace cl = llvm::cl;

cl::OptionCategory
    CompilerCategory("Compiler Options", "Options for controlling the compilation process.");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"),
                                          cl::cat(CompilerCategory));

struct Data {
    Data()
        : fout("../main.mlir") {
        fout << "module {\n";
        fout << "\tfunc.func @compute_graph_forward() {\n";
    }
    std::ofstream fout;
    ~Data() {
        fout << "\t}\n";
        fout << "}\n";
    }
};

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
//
// stablechlo

// TODO: optimize with string_builder
std::string tensor_type_to_mlir(const ggml_tensor* t) {
    std::string result = "tensor<";

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result += std::to_string(t->ne[i]) + 'x';
    }
    result += "f32>";  // TODO: different types

    return result;
}

std::string string_to_variable_name(const char* name) {
    std::string result = "%";
    result.append(name);
    for (int i = 0; i < std::ssize(result); ++i) {
        switch (result[i]) {
            case ' ':
            case '(':
            case ')': {
                result[i] = '_';
                break;
            }
        }
    }
    return result;
}

int main(int argc, char** argv) {
    // cl::HideUnrelatedOptions( CompilerCategory );
    cl::ParseCommandLineOptions(argc, argv);
    std::cout << inputFilename << '\n';
    // llama_model_params model_params = llama_model_default_params();
    // struct llama_model* model = llama_model_load_from_file(argv[1], model_params);
    //
    // llama_context_params ctx_params = llama_context_default_params();
    // Data data{};
    // ctx_params.cb_eval_user_data = &data;
    // ctx_params.cb_eval = [](ggml_tensor* t, bool ask, void* user_data) {
    //     if (ask) {
    //         return true;
    //     }
    //     Data* data = static_cast<Data*>(user_data);
    //
    //     const struct ggml_tensor* src0 = t->src[0];
    //     const struct ggml_tensor* src1 = t->src[1];
    //
    //     if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_ADD) {
    //         std::string linalg_op_name;
    //         if (t->op == GGML_OP_MUL_MAT) {
    //             linalg_op_name = "matmul";
    //         } else {
    //             linalg_op_name = "add";
    //         }
    //         data->fout << "\t\t" << string_to_variable_name(t->name) << "_init = tensor.empty() : " << tensor_type_to_mlir(t) << '\n';  // may be ambiguous, but doesn't matter when move to ssa
    //         data->fout << "\t\t" << string_to_variable_name(t->name) << " = linalg." << linalg_op_name << " ins(";
    //         data->fout << string_to_variable_name(src0->name) << ", " << string_to_variable_name(src1->name) << " : ";
    //         data->fout << tensor_type_to_mlir(src0) << ", " << tensor_type_to_mlir(src1) << ")\n";
    //
    //         data->fout << "\t\t\touts(" << string_to_variable_name(t->name) << "_init : " << tensor_type_to_mlir(t);
    //         data->fout << ") -> " << tensor_type_to_mlir(t) << "\n\n";
    //     }
    //
    //     data->fout << "\t\t" << string_to_variable_name(t->name) << " = ";
    //     data->fout << '\"' << ggml_op_name(t->op) << "\"(" << string_to_variable_name(src0->name);
    //     if (src1) {
    //         data->fout << ", " << string_to_variable_name(src1->name);
    //     }
    //     data->fout << ") : (" << tensor_type_to_mlir(src0);
    //     if (src1) {
    //         data->fout << ", " << tensor_type_to_mlir(src1);
    //     }
    //     data->fout << ") -> " << tensor_type_to_mlir(t) << "\n\n";
    //
    //     return true;
    // };
    //
    // llama_context* ctx = llama_init_from_model(model, ctx_params);
    // std::vector<llama_token> tokens = {1, 2, 3};
    // llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

    return 0;
}
