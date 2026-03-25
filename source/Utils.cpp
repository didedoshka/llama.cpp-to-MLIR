#include "Utils.h"
#include "llvm/Support/DebugLog.h"

float ggmlTensorGet(const ggml_tensor *tensor, size_t i0, size_t i1, size_t i2, size_t i3) {
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
