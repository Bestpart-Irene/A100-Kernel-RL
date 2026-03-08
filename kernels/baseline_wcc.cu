/*
Baseline WCC Implementation - Standard cuGraph-style approach
Reference implementation for comparison with optimized H100 kernels.

This kernel uses traditional atomic operations and multiple launches,
representing the "before" state for optimization.
*/
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Union-Find with path splitting (baseline approach)
__device__ int find_root_atomic(int* parent, int x) {
    while (parent[x] != x) {
        int p = parent[x];
        int gp = parent[p];
        // Path splitting: point x to grandparent
        atomicCAS(&parent[x], p, gp);
        x = p;  // Move to parent (guaranteed progress)
    }
    return x;
}

__global__ void wcc_initialize(int* parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        parent[tid] = tid;  // Each vertex starts as its own component
    }
}

__global__ void wcc_hook_phase(
    int* parent, 
    const int* row_ptr, 
    const int* col_idx, 
    int N,
    bool* changed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    int v = tid;
    int root_v = find_root_atomic(parent, v);
    
    // Process all edges of vertex v
    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        if (u >= N) continue;  // Safety check
        
        int root_u = find_root_atomic(parent, u);
        
        if (root_v != root_u) {
            // Atomic union operation
            int lo = min(root_v, root_u);
            int hi = max(root_v, root_u);
            
            int old_hi = atomicCAS(&parent[hi], hi, lo);
            if (old_hi == hi) {
                *changed = true;
                if (lo == root_u) root_v = lo;  // Update local root if needed
            } else if (old_hi != lo) {
                // CAS failed, retry with new value
                root_v = find_root_atomic(parent, v);
            }
        }
    }
}

__global__ void wcc_compress_phase(int* parent, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    // Final path compression
    int root = find_root_atomic(parent, tid);
    parent[tid] = root;
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
        bool* d_changed;
        
        // Allocate device memory
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_changed, sizeof(bool));
        
        // Copy graph data to device
        int* d_row_ptr;
        int* d_col_idx;
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, (row_ptr[num_vertices]) * sizeof(int));
        
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch configuration
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        // Initialize parent array
        wcc_initialize<<<grid_size, block_size>>>(d_parent, num_vertices);
        cudaDeviceSynchronize();
        
        // Iterative WCC algorithm
        bool h_changed = true;
        int max_iterations = 100;
        int iteration = 0;
        
        while (h_changed && iteration < max_iterations) {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
            
            // Hook phase
            wcc_hook_phase<<<grid_size, block_size>>>(
                d_parent, d_row_ptr, d_col_idx, num_vertices, d_changed
            );
            cudaDeviceSynchronize();
            
            // Check if any changes occurred
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
            
            iteration++;
        }
        
        // Final compression
        wcc_compress_phase<<<grid_size, block_size>>>(d_parent, num_vertices);
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_parent);
        cudaFree(d_changed);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
    }
}
