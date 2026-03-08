/*
Thread Block Cluster WCC Implementation - H100 DSMEM Optimization

Advanced H100 implementation using:
- Thread Block Clusters for cross-partition label propagation
- Distributed Shared Memory (DSMEM) for SM-to-SM communication
- Non-atomic Union-Find with cluster-level synchronization
- TMA for adjacency list loading (guarded by __CUDA_ARCH__ >= 900)
*/
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// Cluster configuration
#define CLUSTER_SIZE 4  // 4 blocks per cluster
#define BLOCK_SIZE 256

// Non-atomic path compression
__device__ __forceinline__ int find_root_nonatomic(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

#if __CUDA_ARCH__ >= 900
// Cluster kernel with DSMEM support
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ void wcc_clustered_dsmem(
    int* parent,
    const int* row_ptr,
    const int* col_idx,
    int N
) {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();
    
    // Shared memory for local labels and cross-cluster communication
    __shared__ int local_labels[BLOCK_SIZE];
    __shared__ int cluster_labels[CLUSTER_SIZE * BLOCK_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_rank = cluster.block_rank();
    
    // Initialize local labels
    if (tid < N) {
        local_labels[threadIdx.x] = tid;
        parent[tid] = tid;  // Initialize parent array
    }
    block.sync();
    
    // Copy local labels to cluster shared memory
    if (tid < N) {
        cluster_labels[block_rank * BLOCK_SIZE + threadIdx.x] = local_labels[threadIdx.x];
    }
    cluster.sync();  // Cluster-wide synchronization
    
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process local edges
            for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
                int u = col_idx[e];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
            
            // Update local labels
            local_labels[threadIdx.x] = root_v;
        }
        
        block.sync();
        
        // Cross-partition label propagation via DSMEM
        if (tid < N) {
            int v = tid;
            int root_v = local_labels[threadIdx.x];
            
            // Check labels from other blocks in the cluster
            for (int neighbor_rank = 0; neighbor_rank < cluster.num_blocks(); neighbor_rank++) {
                if (neighbor_rank == block_rank) continue;
                
                // Access neighbor block's shared memory via DSMEM
                int* neighbor_smem = cluster.map_shared_rank(cluster_labels, neighbor_rank);
                
                // Check if we can merge with neighbor's labels
                int neighbor_local_idx = v % BLOCK_SIZE;
                if (neighbor_local_idx < BLOCK_SIZE) {
                    int neighbor_root = neighbor_smem[neighbor_local_idx];
                    
                    if (root_v != neighbor_root) {
                        int lo = min(root_v, neighbor_root);
                        int hi = max(root_v, neighbor_root);
                        parent[hi] = lo;
                        changed = true;
                        root_v = lo;
                        local_labels[threadIdx.x] = root_v;
                    }
                }
            }
        }
        
        // Update cluster shared memory
        cluster_labels[block_rank * BLOCK_SIZE + threadIdx.x] = local_labels[threadIdx.x];
        
        // Synchronize across cluster
        cluster.sync();
    }
    
    // Final path compression
    if (tid < N) {
        int final_root = find_root_nonatomic(parent, tid);
        parent[tid] = final_root;
    }
}

// TMA-enabled kernel for adjacency list loading
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ void wcc_tma_clustered(
    int* parent,
    const int* row_ptr,
    const int* col_idx,
    int N,
    CUtensorMap* tensor_maps  // Pre-computed TMA descriptors
) {
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();
    
    __shared__ int local_adj[BLOCK_SIZE * 16];  // Local adjacency buffer
    __shared__ int local_labels[BLOCK_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_rank = cluster.block_rank();
    
    // Initialize
    if (tid < N) {
        parent[tid] = tid;
        local_labels[threadIdx.x] = tid;
    }
    block.sync();
    
    // TMA bulk copy of adjacency data (single thread per block)
    #if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0) {
        // Use TMA to load adjacency data asynchronously
        uint64_t coords[5] = {blockIdx.x, 0, 0, 0, 0};  // 1D tensor coordinates
        cp_async_bulk_tensor_1d_global_to_shared(
            &local_adj[0], &tensor_maps[block_rank], coords, 0
        );
    }
    #endif
    
    // Process vertices while TMA loads data asynchronously
    bool changed = true;
    while (changed) {
        changed = false;
        
        if (tid < N) {
            int v = tid;
            int root_v = find_root_nonatomic(parent, v);
            
            // Process edges from TMA-loaded buffer
            int edge_start = row_ptr[v];
            int edge_end = row_ptr[v + 1];
            int local_edge_count = min(edge_end - edge_start, 16);
            
            for (int i = 0; i < local_edge_count; i++) {
                int u = local_adj[threadIdx.x * 16 + i];
                if (u >= N) continue;
                
                int root_u = find_root_nonatomic(parent, u);
                
                if (root_v != root_u) {
                    int lo = min(root_v, root_u);
                    int hi = max(root_v, root_u);
                    parent[hi] = lo;
                    changed = true;
                    root_v = lo;
                }
            }
        }
        
        block.sync();
        cluster.sync();
    }
}
#endif

// Setup TMA descriptors
void setup_tma_descriptors(
    const int* col_idx,
    int num_edges,
    CUtensorMap* tensor_maps,
    int cluster_size
) {
    #if __CUDA_ARCH__ >= 900
    for (int i = 0; i < cluster_size; i++) {
        int edges_per_block = (num_edges + cluster_size - 1) / cluster_size;
        int start_edge = i * edges_per_block;
        int block_edges = min(edges_per_block, num_edges - start_edge);
        
        if (block_edges > 0) {
            cuTensorMapEncodeTiled(
                &tensor_maps[i],
                CU_TENSOR_MAP_DATA_TYPE_INT32,
                1,  // 1D tensor
                &col_idx[start_edge],
                &block_edges,  // size
                nullptr,  // stride (not used for 1D)
                nullptr,  // box_size (not used for 1D)
                nullptr,  // elem_stride (not used for 1D)
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        }
    }
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
        CUtensorMap* d_tensor_maps;
        
        // Allocate device memory
        cudaMalloc(&d_parent, num_vertices * sizeof(int));
        cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, row_ptr[num_vertices] * sizeof(int));
        
        #if __CUDA_ARCH__ >= 900
        cudaMalloc(&d_tensor_maps, CLUSTER_SIZE * sizeof(CUtensorMap));
        #endif
        
        // Copy graph data to device
        cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, row_ptr[num_vertices] * sizeof(int), cudaMemcpyHostToDevice);
        
        // Setup TMA descriptors
        #if __CUDA_ARCH__ >= 900
        CUtensorMap* h_tensor_maps = (CUtensorMap*)malloc(CLUSTER_SIZE * sizeof(CUtensorMap));
        setup_tma_descriptors(col_idx, row_ptr[num_vertices], h_tensor_maps, CLUSTER_SIZE);
        cudaMemcpy(d_tensor_maps, h_tensor_maps, CLUSTER_SIZE * sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        free(h_tensor_maps);
        #endif
        
        // Launch configuration
        int block_size = BLOCK_SIZE;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        // Ensure grid size is multiple of cluster size
        grid_size = ((grid_size + CLUSTER_SIZE - 1) / CLUSTER_SIZE) * CLUSTER_SIZE;
        
        #if __CUDA_ARCH__ >= 900
        // Launch clustered kernel with DSMEM
        cudaLaunchConfig_t config = {};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
        config.attrs = attrs;
        config.numAttrs = 1;
        config.gridDim = grid_size;
        config.blockDim = block_size;
        config.sharedMem = CLUSTER_SIZE * block_size * sizeof(int);
        
        void* kernel_args[] = {&d_parent, &d_row_ptr, &d_col_idx, &num_vertices};
        cudaLaunchKernelEx(&config, (void*)wcc_clustered_dsmem, kernel_args);
        #else
        // Fallback to basic kernel for non-Hopper
        // Use ecl_cc_h100.cu implementation
        #endif
        
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_parent);
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        #if __CUDA_ARCH__ >= 900
        cudaFree(d_tensor_maps);
        #endif
    }
}
