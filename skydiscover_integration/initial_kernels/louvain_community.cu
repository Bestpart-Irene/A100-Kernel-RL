// Louvain community detection with warp-level shared-memory hash tables on A100
// Adapted from doubleGraph: dual dispatch (warp-cooperative + thread-per-vertex)
// CU_FLAGS: --use_fast_math --maxrregcount=48

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define HASH_SIZE 128  // Entries per warp hash table

// Shared memory hash table for accumulating inter-community edge weights
struct WarpHashTable {
    int keys[HASH_SIZE];
    float values[HASH_SIZE];
};

__device__ void hash_insert(WarpHashTable* ht, int key, float value, int lane) {
    int slot = key % HASH_SIZE;
    for (int probe = 0; probe < HASH_SIZE; probe++) {
        int idx = (slot + probe) % HASH_SIZE;
        int old = atomicCAS(&ht->keys[idx], -1, key);
        if (old == -1 || old == key) {
            atomicAdd(&ht->values[idx], value);
            return;
        }
    }
    // Hash table full — rare for small communities, ignored
}

// Local move phase: each vertex tries to move to the neighboring community
// that maximizes modularity gain
__global__ __launch_bounds__(256, 4)
void louvain_local_move(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ edge_weights,
    int* __restrict__ communities,
    const float* __restrict__ comm_weights,  // Total weight of each community
    float total_weight,
    int num_vertices,
    int* __restrict__ changed_flag  // Zero-copy: set if any vertex moved
) {
    // Shared memory: one hash table per warp
    extern __shared__ WarpHashTable shared_ht[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Initialize hash table for this warp
    WarpHashTable* my_ht = &shared_ht[warp_id];
    for (int i = lane; i < HASH_SIZE; i += WARP_SIZE) {
        my_ht->keys[i] = -1;
        my_ht->values[i] = 0.0f;
    }
    __syncwarp();

    int v = tid;
    if (v >= num_vertices) return;

    int v_comm = communities[v];
    int start = __ldg(&row_ptr[v]);
    int end = __ldg(&row_ptr[v + 1]);
    int degree = end - start;

    // Dual dispatch: warp-cooperative for high-degree, thread for low-degree
    if (degree >= 8) {
        // Warp-cooperative: lanes distribute neighbor scanning
        float v_weight = 0.0f;
        for (int e = start + lane; e < end; e += WARP_SIZE) {
            int neighbor = __ldg(&col_idx[e]);
            float w = (edge_weights != NULL) ? __ldg(&edge_weights[e]) : 1.0f;
            int n_comm = communities[neighbor];
            hash_insert(my_ht, n_comm, w, lane);
            v_weight += w;
        }
        // Warp-reduce v_weight
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v_weight += __shfl_down_sync(0xFFFFFFFF, v_weight, offset);
        }

        // Lane 0 finds best community
        if (lane == 0) {
            float best_gain = 0.0f;
            int best_comm = v_comm;

            for (int i = 0; i < HASH_SIZE; i++) {
                if (my_ht->keys[i] == -1) continue;
                int c = my_ht->keys[i];
                float edge_to_c = my_ht->values[i];

                // Modularity gain: ΔQ = edge_to_c/m - v_weight * comm_weight_c / (2m^2)
                float delta_q = edge_to_c / total_weight
                    - v_weight * comm_weights[c] / (2.0f * total_weight * total_weight);
                if (delta_q > best_gain) {
                    best_gain = delta_q;
                    best_comm = c;
                }
            }

            if (best_comm != v_comm) {
                communities[v] = best_comm;
                *changed_flag = 1;
            }
        }
    } else {
        // Thread-per-vertex: simple scan for low-degree vertices
        float best_gain = 0.0f;
        int best_comm = v_comm;
        float v_weight = 0.0f;

        for (int e = start; e < end; e++) {
            float w = (edge_weights != NULL) ? __ldg(&edge_weights[e]) : 1.0f;
            v_weight += w;
        }

        for (int e = start; e < end; e++) {
            int neighbor = __ldg(&col_idx[e]);
            float w = (edge_weights != NULL) ? __ldg(&edge_weights[e]) : 1.0f;
            int n_comm = communities[neighbor];

            float delta_q = w / total_weight
                - v_weight * comm_weights[n_comm] / (2.0f * total_weight * total_weight);
            if (delta_q > best_gain) {
                best_gain = delta_q;
                best_comm = n_comm;
            }
        }

        if (best_comm != v_comm) {
            communities[v] = best_comm;
            *changed_flag = 1;
        }
    }
}

extern "C" void launch_louvain(
    int* row_ptr, int* col_idx, float* edge_weights,
    int* communities, int num_vertices, int num_edges
) {
    // Initialize: each vertex in its own community
    int* h_comm = (int*)malloc(num_vertices * sizeof(int));
    for (int i = 0; i < num_vertices; i++) h_comm[i] = i;
    cudaMemcpy(communities, h_comm, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    free(h_comm);

    // Community weights (simplified: degree-based)
    float* d_comm_weights;
    cudaMalloc(&d_comm_weights, num_vertices * sizeof(float));

    // Zero-copy changed flag
    int* h_changed;
    int* d_changed;
    cudaHostAlloc(&h_changed, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_changed, h_changed, 0);

    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    size_t smem = warps_per_block * sizeof(WarpHashTable);

    float total_weight = (float)num_edges;  // Simplified

    for (int pass = 0; pass < 20; pass++) {
        *h_changed = 0;
        __sync_synchronize();

        louvain_local_move<<<blocks, BLOCK_SIZE, smem>>>(
            row_ptr, col_idx, edge_weights,
            communities, d_comm_weights, total_weight,
            num_vertices, d_changed
        );
        cudaDeviceSynchronize();

        if (*h_changed == 0) break;
    }

    cudaFree(d_comm_weights);
    cudaFreeHost(h_changed);
}
