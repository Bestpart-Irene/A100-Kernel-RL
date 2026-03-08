// Direction-optimizing BFS for power-law graphs on A100
// Adapted from doubleGraph: bitmap frontier + top-down/bottom-up switch
// CU_FLAGS: --use_fast_math --extra-device-vectorization

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CSR graph structure
struct CSRGraph {
    int* row_ptr;
    int* col_idx;
    int num_vertices;
    int num_edges;
};

// Bitmap frontier for dense frontier representation
__device__ unsigned int* d_bitmap_frontier;
__device__ unsigned int* d_bitmap_next;
__device__ int* d_frontier_queue;
__device__ int* d_queue_size;

__global__ __launch_bounds__(256, 4)
void bfs_top_down(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ distances,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_size,
    int current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int v = frontier[tid];
    int start = __ldg(&row_ptr[v]);
    int end = __ldg(&row_ptr[v + 1]);

    for (int e = start; e < end; e++) {
        int neighbor = __ldg(&col_idx[e]);
        if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void bfs_bottom_up(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ distances,
    const unsigned int* __restrict__ frontier_bitmap,
    unsigned int* __restrict__ next_bitmap,
    int* __restrict__ next_count,
    int num_vertices,
    int current_level
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (distances[v] != -1) return;  // Already visited

    int start = __ldg(&row_ptr[v]);
    int end = __ldg(&row_ptr[v + 1]);

    for (int e = start; e < end; e++) {
        int neighbor = __ldg(&col_idx[e]);
        // Check if neighbor is in current frontier via bitmap
        unsigned int word = frontier_bitmap[neighbor / 32];
        if (word & (1u << (neighbor % 32))) {
            distances[v] = current_level + 1;
            // Add to next frontier bitmap
            atomicOr(&next_bitmap[v / 32], 1u << (v % 32));
            atomicAdd(next_count, 1);
            break;
        }
    }
}

extern "C" void launch_bfs(
    int* row_ptr, int* col_idx, int* distances,
    int num_vertices, int num_edges, int source
) {
    // Allocate frontiers
    int* d_frontier1;
    int* d_frontier2;
    int* d_next_size;
    unsigned int* d_bitmap1;
    unsigned int* d_bitmap2;
    int bitmap_words = (num_vertices + 31) / 32;

    cudaMalloc(&d_frontier1, num_vertices * sizeof(int));
    cudaMalloc(&d_frontier2, num_vertices * sizeof(int));
    cudaMalloc(&d_next_size, sizeof(int));
    cudaMalloc(&d_bitmap1, bitmap_words * sizeof(unsigned int));
    cudaMalloc(&d_bitmap2, bitmap_words * sizeof(unsigned int));

    // Initialize distances to -1, source to 0
    cudaMemset(distances, 0xFF, num_vertices * sizeof(int));
    int zero = 0;
    cudaMemcpy(&distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier1, &source, sizeof(int), cudaMemcpyHostToDevice);

    int frontier_size = 1;
    int level = 0;
    // Direction-optimizing threshold: switch to bottom-up when frontier > V/10
    int bu_threshold = num_vertices / 10;

    while (frontier_size > 0) {
        cudaMemset(d_next_size, 0, sizeof(int));

        if (frontier_size < bu_threshold) {
            // Top-down: queue-based, good for sparse frontiers
            int blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bfs_top_down<<<blocks, BLOCK_SIZE>>>(
                row_ptr, col_idx, distances,
                d_frontier1, frontier_size, d_frontier2, d_next_size, level
            );
        } else {
            // Bottom-up: bitmap scan, good for dense frontiers (power-law expansion)
            cudaMemset(d_bitmap2, 0, bitmap_words * sizeof(unsigned int));
            int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            bfs_bottom_up<<<blocks, BLOCK_SIZE>>>(
                row_ptr, col_idx, distances,
                d_bitmap1, d_bitmap2, d_next_size, num_vertices, level
            );
            // Swap bitmaps
            unsigned int* tmp = d_bitmap1;
            d_bitmap1 = d_bitmap2;
            d_bitmap2 = tmp;
        }

        cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap frontier buffers
        int* tmp = d_frontier1;
        d_frontier1 = d_frontier2;
        d_frontier2 = tmp;
        level++;
    }

    cudaFree(d_frontier1);
    cudaFree(d_frontier2);
    cudaFree(d_next_size);
    cudaFree(d_bitmap1);
    cudaFree(d_bitmap2);
}
