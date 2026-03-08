/*
 * Seed kernel: doubleGraph WCC adapted for KernelForge evaluation interface.
 * Original: doubleGraph/cpp/src/aai/impl/a100/components/weakly_connected_components.cu
 * License: Apache 2.0 (AA-I Technologies Ltd.)
 *
 * Key A100 optimizations:
 * - Zero-copy convergence flag (cudaHostAlloc + cudaHostGetDevicePointer)
 * - Path-halving Union-Find (non-atomic path compression)
 * - __launch_bounds__(256) for register control
 * - Sampled hooking phase before full scan
 * - __ldg() for read-only graph data
 *
 * // CU_FLAGS: --use_fast_math --extra-device-vectorization
 */
#include <cuda_runtime.h>
#include <cstdint>

// --- Device kernels ---

__global__ __launch_bounds__(256)
void dg_wcc_init(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        parent[i] = i;
    }
}

__device__ __forceinline__ int dg_find_root(int* __restrict__ parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        int gp = parent[p];
        parent[v] = gp;   // Path halving — no atomics
        v = p;
        p = gp;
    }
    return p;
}

__device__ __forceinline__ void dg_link(int* __restrict__ parent, int u, int v) {
    int ru = dg_find_root(parent, u);
    int rv = dg_find_root(parent, v);

    while (ru != rv) {
        int hi = ru > rv ? ru : rv;
        int lo = ru < rv ? ru : rv;

        int old = atomicCAS(&parent[hi], hi, lo);
        if (old == hi) break;

        ru = dg_find_root(parent, u);
        rv = dg_find_root(parent, v);
    }
}

// Sampled hooking: only process first k neighbors per vertex (fast convergence for hubs)
__global__ __launch_bounds__(256)
void dg_wcc_hook_sample(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int k
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n;
         u += blockDim.x * gridDim.x) {
        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);
        int degree = end - start;
        int samples = degree < k ? degree : k;

        for (int s = 0; s < samples; s++) {
            int v = __ldg(&indices[start + s]);
            dg_link(parent, u, v);
        }
    }
}

// Full hooking: process all edges
__global__ __launch_bounds__(256)
void dg_wcc_hook_full(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int skip_root
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n;
         u += blockDim.x * gridDim.x) {
        if (dg_find_root(parent, u) == skip_root) continue;

        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);

        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            dg_link(parent, u, v);
        }
    }
}

// Path compression
__global__ __launch_bounds__(256)
void dg_wcc_compress(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        int v = i;
        int p = parent[v];
        while (p != parent[p]) {
            int gp = parent[p];
            parent[v] = gp;
            v = p;
            p = gp;
        }
        parent[i] = p;
    }
}

// Shortcut + convergence check (writes to zero-copy flag)
__global__ __launch_bounds__(256)
void dg_wcc_shortcut_check(int* __restrict__ parent, int n, int* __restrict__ not_converged) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) {
            parent[i] = gp;
            *not_converged = 1;
        }
    }
}

__global__ __launch_bounds__(256)
void dg_wcc_shortcut(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) parent[i] = gp;
    }
}

// --- Host interface matching KernelForge evaluation ---

extern "C" {

void wcc_kernel(
    const int* row_ptr,
    const int* col_idx,
    int num_vertices,
    int* labels
) {
    if (num_vertices == 0) return;

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;

    // Allocate device memory
    int* d_parent;
    int* d_row_ptr;
    int* d_col_idx;
    int num_edges = row_ptr[num_vertices];

    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    // Zero-copy convergence flag
    int* h_flag;
    int* d_flag;
    cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_flag, h_flag, 0);

    // Phase 1: Initialize
    dg_wcc_init<<<grid, BLOCK>>>(d_parent, num_vertices);

    // Phase 2: Sampled hooking (k=2) — fast convergence for hub vertices
    dg_wcc_hook_sample<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_parent, num_vertices, 2);

    // Phase 3: Compress
    dg_wcc_compress<<<grid, BLOCK>>>(d_parent, num_vertices);

    // Phase 4: Full hooking (skip already-connected root)
    dg_wcc_hook_full<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_parent, num_vertices, 0);

    // Phase 5: Compress again
    dg_wcc_compress<<<grid, BLOCK>>>(d_parent, num_vertices);

    // Phase 6: Iterative shortcut until convergence (zero-copy check)
    *h_flag = 0;
    __sync_synchronize();
    dg_wcc_shortcut_check<<<grid, BLOCK>>>(d_parent, num_vertices, d_flag);
    cudaDeviceSynchronize();

    if (*h_flag) {
        for (int iter = 0; iter < 100; iter++) {
            for (int j = 0; j < 5; j++) {
                dg_wcc_shortcut<<<grid, BLOCK>>>(d_parent, num_vertices);
            }
            *h_flag = 0;
            __sync_synchronize();
            dg_wcc_shortcut_check<<<grid, BLOCK>>>(d_parent, num_vertices, d_flag);
            cudaDeviceSynchronize();
            if (!(*h_flag)) break;
        }
    }

    // Copy results back
    cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_parent);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFreeHost(h_flag);
}

}  // extern "C"
