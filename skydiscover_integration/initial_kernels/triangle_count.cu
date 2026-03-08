// Triangle counting with warp-level set intersection for community-structured graphs on A100
// Adapted from doubleGraph: sorted merge intersection + warp-cooperative processing
// CU_FLAGS: --use_fast_math --maxrregcount=64

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level binary search in sorted neighbor list
__device__ int warp_binary_search(
    const int* __restrict__ col_idx,
    int start, int end, int target
) {
    while (start < end) {
        int mid = (start + end) / 2;
        int val = __ldg(&col_idx[mid]);
        if (val == target) return 1;
        if (val < target) start = mid + 1;
        else end = mid;
    }
    return 0;
}

// Per-vertex triangle counting using sorted merge intersection
// Each warp handles one vertex: lanes cooperatively search neighbor intersections
__global__ __launch_bounds__(256, 4)
void count_triangles_warp(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    unsigned long long* __restrict__ triangle_count,
    int num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_vertices) return;

    int v = warp_id;
    int v_start = __ldg(&row_ptr[v]);
    int v_end = __ldg(&row_ptr[v + 1]);
    int v_degree = v_end - v_start;

    unsigned long long local_count = 0;

    // For each neighbor u of v (distributed across lanes)
    for (int i = lane; i < v_degree; i += WARP_SIZE) {
        int u = __ldg(&col_idx[v_start + i]);
        if (u <= v) continue;  // Only count u > v to avoid double-counting

        int u_start = __ldg(&row_ptr[u]);
        int u_end = __ldg(&row_ptr[u + 1]);

        // Sorted merge intersection of N(v) and N(u)
        int p = v_start;
        int q = u_start;
        while (p < v_end && q < u_end) {
            int a = __ldg(&col_idx[p]);
            int b = __ldg(&col_idx[q]);
            if (a == b) {
                if (a > u) {  // w > u > v ordering
                    local_count++;
                }
                p++;
                q++;
            } else if (a < b) {
                p++;
            } else {
                q++;
            }
        }
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }

    // Lane 0 writes the result
    if (lane == 0 && local_count > 0) {
        atomicAdd(triangle_count, local_count);
    }
}

extern "C" unsigned long long launch_triangle_count(
    int* row_ptr, int* col_idx, int num_vertices, int num_edges
) {
    unsigned long long* d_count;
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    // One warp per vertex
    int total_threads = num_vertices * WARP_SIZE;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    count_triangles_warp<<<blocks, BLOCK_SIZE>>>(
        row_ptr, col_idx, d_count, num_vertices
    );

    unsigned long long h_count;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_count;
}
