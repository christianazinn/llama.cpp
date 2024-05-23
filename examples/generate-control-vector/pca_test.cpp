// only necessary for testing
#include <iostream>
//
#include <random>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <ctime>

// Function to perform power iteration to find the principal eigenvector
static struct ggml_tensor* power_iteration(struct ggml_tensor* mat, int maxIterations = 1000, float tolerance = 1e-8) {

    // get dimensions 
    // TODO are we sure we're always getting a square?
    int n = mat->ne[0];

    // create the random vector
    // TODO figure out how to calculate memory allocation necessary
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor* b_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // random vector generation
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        ggml_set_f32_1d(b_tensor, i, distribution(generator));
    }

    // random vector normalization
    // FIXME
    b_tensor = ggml_norm_inplace(ctx, b_tensor, 1e-7);

    for (int i = 0; i < n; i++) {
        float val = ggml_get_f32_1d(b_tensor, i);
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "-------------------" << std::endl;

    for (int iter = 0; iter < maxIterations; ++iter) {

        // store the previous one so we can check for convergence
        struct ggml_tensor* b_prev_tensor = ggml_dup_tensor(ctx, b_tensor);

        // matrix multiplication and renormalize
        // FIXME
        b_tensor = ggml_mul_mat(ctx, mat, b_tensor);

        for (int i = 0; i < n; i++) {
            float val = ggml_get_f32_1d(b_tensor, i);
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        // FIXME
        b_tensor = ggml_norm_inplace(ctx, b_tensor, 1);

        // convergence check
        float diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff += std::pow(ggml_get_f32_1d(b_prev_tensor, i) - ggml_get_f32_1d(b_tensor, i), 2);
        }
        if (std::sqrt(diff) < tolerance) {
            break;
        }
    }

    return b_tensor;
}

// TESTING CODE

int main() {
    int n = 4;
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create a dense matrix (4x4) for testing
    std::vector<float> mat_data = {
        4, 2, 0, 0,
        1, 3, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 1
    };
    struct ggml_tensor* mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);

    // put the data in the tensor
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ggml_set_f32_nd(mat, i, j, 0, 0, mat_data[i*n + j]);
        }
    }

    // Perform power iteration to find the dominant eigenvector
    struct ggml_tensor* dominantEigenvector = power_iteration(mat);

    // Output the dominant eigenvector
    std::cout << "Dominant Eigenvector: \n";
    for (int i = 0; i < n; i++) {
        float val = ggml_get_f32_1d(dominantEigenvector, i);
        std::cout << val << " ";
    }
    std::cout << std::endl;

    ggml_free(ctx);
    return 0;
}
