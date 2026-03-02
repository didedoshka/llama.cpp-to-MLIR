#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define LOG printf

static void ggml_print_tensor(ggml_tensor* tensor, int64_t n) {
    const int64_t* ne = tensor->ne;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(" ..., \n");
                i2 = ne[2] - n;
            }
            LOG(" [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG("  ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("  [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = ggml_get_f32_nd(tensor, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1)
                        LOG(", ");
                }
                LOG("],\n");
            }
            LOG(" ],\n");
        }
        LOG("]\n");
    }
}

int main(void) {
    ggml_time_init();

    const int a_ne0 = 4, a_ne1 = 3, a_ne2 = 2;

    float a_matrix[a_ne0 * a_ne1 * a_ne2];
    for (int i = 0; i < a_ne0 * a_ne1 * a_ne2; ++i) {
        a_matrix[i] = i + 1;
    }

    const int ne0 = 2, ne1 = 2;
    int b_matrix[ne0 * ne0] = {
        2, 0, 1, 2,
    };

    struct ggml_init_params params{
        /*.mem_size   =*/1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };

    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a_ne0, a_ne1, a_ne2);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne0, ne1);
    memcpy(a->data, a_matrix, ggml_nbytes(a));
    memcpy(b->data, b_matrix, ggml_nbytes(b));

    struct ggml_tensor* result = ggml_get_rows(ctx, a, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    ggml_print_tensor(a, 10);
    ggml_print_tensor(b, 10);
    ggml_print_tensor(result, 10);

    return 0;
}
