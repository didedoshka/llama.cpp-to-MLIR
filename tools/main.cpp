
#include <ggml-cpp.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  llama_model_params model_params = llama_model_default_params();
  struct llama_model *model = llama_model_load_from_file(argv[1], model_params);

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.cb_eval = [](ggml_tensor *tensor, bool ask, void *user_data) {
    std::cout << "eval [" << ggml_op_name(tensor->op) << "]" << std::endl;
    return true;
  };

  /// @todo: Это пример использования callback-ов.
  // С помощью этой возможности можно экспортировать модель в mlir.
  llama_context *ctx = llama_init_from_model(model, ctx_params);
  std::vector<llama_token> tokens = {1, 2, 3};
  llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
  return 0;
}
