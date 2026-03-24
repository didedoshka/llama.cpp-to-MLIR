#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <string>


int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(argv[1], model_params);

    llama_context_params ctx_params = llama_context_default_params();
    // Data data{};
    ctx_params.cb_eval_user_data = nullptr;
    ctx_params.cb_eval = [](ggml_tensor* t, bool ask, void* user_data) {
        if (ask) {
            return true;
        }
        std::cout << ggml_op_name(t->op) << '\n';
        // Data* data = static_cast<Data*>(user_data);

        // const struct ggml_tensor* src0 = t->src[0];
        // const struct ggml_tensor* src1 = t->src[1];

        // data->fout << "\t\t" << string_to_variable_name(t->name) << " = ";
        // data->fout << '\"' << ggml_op_name(t->op) << "\"(" << string_to_variable_name(src0->name);
        // if (src1) {
        //     data->fout << ", " << string_to_variable_name(src1->name);
        // }
        // data->fout << ") : (" << tensor_type_to_mlir(src0);
        // if (src1) {
        //     data->fout << ", " << tensor_type_to_mlir(src1);
        // }
        // data->fout << ") -> " << tensor_type_to_mlir(t) << "\n\n";

        return true;
    };

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    std::vector<llama_token> tokens = {1, 2, 3};
    llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));

    return 0;
}
