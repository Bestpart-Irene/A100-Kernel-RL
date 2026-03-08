// PageRank with zero-copy convergence check for dense regular graphs on A100
// Adapted from doubleGraph: pinned memory convergence + SpMV
// CU_FLAGS: --use_fast_math

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define DAMPING 0.85f
#define EPSILON 1e-6f
#define MAX_ITERS 100

__global__ __launch_bounds__(256, 4)
void pagerank_spmv(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ rank_in,
    float* __restrict__ rank_out,
    const int* __restrict__ out_degree,
    int num_vertices,
    float damping
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = __ldg(&row_ptr[v]);
    int end = __ldg(&row_ptr[v + 1]);

    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        int neighbor = __ldg(&col_idx[e]);
        int deg = __ldg(&out_degree[neighbor]);
        if (deg > 0) {
            sum += __ldg(&rank_in[neighbor]) / (float)deg;
        }
    }

    rank_out[v] = (1.0f - damping) / (float)num_vertices + damping * sum;
}

__global__ __launch_bounds__(256, 4)
void check_convergence(
    const float* __restrict__ rank_old,
    const float* __restrict__ rank_new,
    int num_vertices,
    int* __restrict__ converged_flag  // Zero-copy host-mapped pointer
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    float diff = fabsf(rank_new[v] - rank_old[v]);
    if (diff > EPSILON) {
        // At least one vertex hasn't converged — clear the flag
        *converged_flag = 0;
    }
}

extern "C" void launch_pagerank(
    int* row_ptr, int* col_idx, int* out_degree,
    float* rank_output, int num_vertices, int num_edges
) {
    float* d_rank_a;
    float* d_rank_b;
    cudaMalloc(&d_rank_a, num_vertices * sizeof(float));
    cudaMalloc(&d_rank_b, num_vertices * sizeof(float));

    // Initialize uniform rank
    float init_rank = 1.0f / num_vertices;
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Use cudaHostAlloc for zero-copy convergence flag
    int* h_converged;
    int* d_converged;
    cudaHostAlloc(&h_converged, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_converged, h_converged, 0);

    // Init ranks on host, copy to device
    float* h_init = (float*)malloc(num_vertices * sizeof(float));
    for (int i = 0; i < num_vertices; i++) h_init[i] = init_rank;
    cudaMemcpy(d_rank_a, h_init, num_vertices * sizeof(float), cudaMemcpyHostToDevice);
    free(h_init);

    float* d_in = d_rank_a;
    float* d_out = d_rank_b;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        // SpMV: rank_out = (1-d)/N + d * A * rank_in / degree
        pagerank_spmv<<<blocks, BLOCK_SIZE>>>(
            row_ptr, col_idx, d_in, d_out, out_degree,
            num_vertices, DAMPING
        );

        // Zero-copy convergence check — no cudaMemcpy needed
        *h_converged = 1;  // Assume converged
        __sync_synchronize();
        check_convergence<<<blocks, BLOCK_SIZE>>>(
            d_in, d_out, num_vertices, d_converged
        );
        cudaDeviceSynchronize();

        if (*h_converged) {
            break;
        }

        // Swap buffers
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // Copy result
    cudaMemcpy(rank_output, d_out, num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rank_a);
    cudaFree(d_rank_b);
    cudaFreeHost(h_converged);
}
