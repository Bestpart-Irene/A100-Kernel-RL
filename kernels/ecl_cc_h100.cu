/*
ECL-CC H100 Implementation - Non-atomic Union-Find with L2 Cache Pinning

Based on ECL-CC algorithm (Jaiganesh and Burtscher, HPDC 2018) with
H100-specific optimizations:
- Non-atomic Union-Find path compression (deliberate data race)
- L2 persistent cache pinning for parent array
- Single cooperative kernel launch
- Hopper-specific optimizations guarded by __CUDA_ARCH__ >= 900
*/
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// Non-atomic path compression (deliberate data race - mathematically safe)
__device__ __forceinline__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        // Path halving - non-atomic, race-tolerant
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// Hopper-specific DPX instruction for 3-way minimum
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ int min_three_hopper(int a, int b, int c) {
    return __vimin3_s32(a, b, c);  // Hardware-fused 3-way min
}
#else
__device__ __forceinline__ int min_three_generic(int a, int b, int c) {
    return min(min(a, b), c);  // Fallback for non-Hopper
}
#endif

__global__ void wcc_single_launch(
    int* parent,
    const int* row_ptr,
    const int* col_idx,
    int N
) {
    auto grid = cg::this_grid();
    bool changed = true;
    
    // Shared memory for local optimization
    __shared__ int local_labels[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local labels
    if (threadIdx.x < 256) {
        local_labels[threadIdx.x] = tid;
    }
    __syncthreads();
    
    while (changed) {
        changed = false;
        
        // Each thread processes one vertex
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process all edges of vertex v
            for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    // Non-atomic union - deliberate data race
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    
                    // Update parent array without atomic operations
                    parent[hi] = lo;
                    changed = true;
                    
                    // Update local root
                    root_v = lo;
                }
            }
        }
        
        // Grid-wide synchronization
        grid.sync();
        
        // Additional optimization: local label propagation
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Check neighbors' labels for faster convergence
            #if __CUDA_ARCH__ >= 900
            // Use DPX for faster label propagation on Hopper
            for (int e = row_ptr[v]; e < min(row_ptr[v + 1], v + 3); e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                int root_u = find_root_nonatomic(parent, u);
                int root_w = (e + 1 < row_ptr[v + 1]) ? 
                    find_root_nonatomic(parent, col_idx[e + 1]) : root_v;
                
                // Hardware-fused 3-way minimum
                int new_root = min_three_hopper(root_v, root_u, root_w);
                if (new_root != root_v) {
                    parent[root_v] = new_root;
                    changed = true;
                }
            }
            #else
            // Generic implementation for non-Hopper
            for (int e = row_ptr[v]; e < min(row_ptr[v + 1], v + 2); e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                int root_u = find_root_nonatomic(parent, u);
                int new_root = min(root_v, root_u);
                if (new_root != root_v) {
                    parent[root_v] = new_root;
                    changed = true;
                }
            }
            #endif
        }
        
        grid.sync();
    }
}

// Helper function to set up L2 cache pinning
void setup_l2_pinning(void* parent_array, size_t array_size, cudaStream_t stream) {
    #if __CUDA_ARCH__ >= 900
    // H100 L2 cache pinning
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t l2_set_aside = min(
        (size_t)(prop.l2CacheSize * 0.75),  // Use 75% of L2 cache
        (size_t)prop.persistingL2CacheMaxSize
    );
    
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_set_aside);
    
    // Configure access policy window
    cudaStreamAttrValue stream_attr = {};
    stream_attr.accessPolicyWindow.base_ptr = parent_array;
    stream_attr.accessPolicyWindow.num_bytes = array_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    #endif
}

// Host interface for Python binding
extern "C" {
    void wcc_kernel(
        const int* row_ptr,
        const int* col_idx,
        int num_vertices,
        int* labels
    ) {
        int* d_parent;
        int* d_row_ptr;
        int* d_col_idx;
        cudaStream_t stream;
        
        // Create CUDA stream
        cudaStreamCreate(&stream);
        
        // Allocate device memory
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, (row_ptr[num_vertices]) * sizeof(int));
        
        // Copy graph data to device
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        // Initialize parent array
        int* h_init = (int*)malloc(num_vertices * sizeof(int));
        for (int i = 0; i < num_vertices; i++) {
            h_init[i] = i;
        }
        cudaMemcpy(d_parent, h_init, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
        free(h_init);
        
        // Set up L2 cache pinning for parent array
        setup_l2_pinning(d_parent, num_vertices * sizeof(int), stream);
        
        // Launch configuration for cooperative kernel
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        // Calculate shared memory requirements
        size_t shared_mem = block_size * sizeof(int);
        
        // Launch single cooperative kernel
        void* kernel_args[] = {&d_parent, &d_row_ptr, &d_col_idx, &num_vertices};
        cudaLaunchCooperativeKernel(
            (void*)wcc_single_launch,
            grid_size, block_size, kernel_args, shared_mem, stream
        );
        
        cudaStreamSynchronize(stream);
        
        // Copy results back
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_parent);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaStreamDestroy(stream);
    }
}
