# SKILLS.md — DoubleGraph: Complete Engineering & Replication Guide

> **Purpose:** This document is written for senior systems engineers, AI coding agents, and individual contributors. It explains, from absolute first principles and at a low technical level, exactly how DoubleGraph successfully replaces generic NVIDIA cuGraph algorithms with hyper-optimized, GPU-specific CUDA kernels. 
> 
> It covers every architectural decision, algorithmic divergence, memory management strategy, and build system quirk required to replicate this system for future hardware (like H100 or B200).

---

## Table of Contents

1. [Problem Statement: The Cost of Genericism](#1-problem-statement-the-cost-of-genericism)
2. [System Architecture: The 4-Layer Drop-in Mechanism](#2-system-architecture-the-4-layer-drop-in-mechanism)
3. [Layer 1: The Graph Abstraction (`compact_graph_t`)](#3-layer-1-the-graph-abstraction-compact_graph_t)
4. [Layer 2: Zero-Overhead Resource Management (`CachePool`)](#4-layer-2-zero-overhead-resource-management-cachepool)
5. [Layer 3: The 4-Way Dispatch Execution Matrix](#5-layer-3-the-4-way-dispatch-execution-matrix)
6. [Layer 4: A100-Specific Kernel Deep Dive](#6-layer-4-a100-specific-kernel-deep-dive)
   - 6.1 [BFS: Dual-Frontier & Direction Optimization](#61-bfs-dual-frontier--direction-optimization)
   - 6.2 [Louvain: 3-Tier Adaptive Dispatch & Shared-Memory Hashing](#62-louvain-3-tier-adaptive-dispatch--shared-memory-hashing)
   - 6.3 [PageRank: Fused SpMV & Warp-Level Reduction](#63-pagerank-fused-spmv--warp-level-reduction)
   - 6.4 [WCC: Parallel Union-Find & Host-Mapped Flags](#64-wcc-parallel-union-find--host-mapped-flags)
   - 6.5 [Triangle Count: DAG Orientation & Bilateral Intersection](#65-triangle-count-dag-orientation--bilateral-intersection)
7. [Cross-Architecture Algorithmic Divergence (A100 vs L4 vs A10G)](#7-cross-architecture-algorithmic-divergence-a100-vs-l4-vs-a10g)
8. [The Integration Layer & Build Pipeline](#8-the-integration-layer--build-pipeline)
9. [Hard Constraints & Known Limitations](#9-hard-constraints--known-limitations)
10. [Porting to Future Hardware: H100 (SM90) and B200 (SM100+)](#10-porting-to-future-hardware-h100-sm90-and-b200-sm100)
11. [Transferable Engineering Strategies](#11-transferable-engineering-strategies)

---

## 1. Problem Statement: The Cost of Genericism

NVIDIA’s cuGraph is built on a highly generic C++ template architecture (`graph_view_t`). It is designed to support single-GPU, multi-GPU, 32-bit/64-bit indices, and hypersparse (DCS) formats within a single algorithmic implementation.

**The Engineering Problem:**
While maintainable, this genericism creates a fundamental performance ceiling. A single CUDA kernel compiled for "any NVIDIA GPU" inherently optimizes for none of them. 

1.  **Memory Hierarchy Disparities:** An A100 (SM80) has **40MB** of L2 cache. An L4 (SM89) has **48MB**. An A10G (SM86) has only **6MB**. A kernel that caches a 1M-vertex state bitmap perfectly in an A100's L2 will cause catastrophic memory thrashing on an A10G.
2.  **PCIe / Launch Overhead Constraints:** On an A10G, lower global memory bandwidth makes host-to-device synchronization proportionally much more expensive. Algorithms must batch operations deeply on the GPU to avoid CPU-GPU roundtrips.
3.  **Execution Width:** An A100 has 108 SMs. An L4 has 58. The mathematical thresholds where an algorithm should switch from a thread-per-vertex strategy to a warp-per-vertex strategy change drastically based on raw compute width.
4.  **Register Pressure:** The optimal `--maxrregcount` varies drastically per architecture. CuGraph applies global CMake flags; we need per-kernel, per-GPU control.

**The Solution:**
DoubleGraph intercepts cuGraph API calls at the lowest C++ template layer and routes them to **Architecture-Aware Implementations (AAI)**—completely distinct, hand-written CUDA kernels tuned specifically for the target GPU's physical constraints.

---

## 2. System Architecture: The 4-Layer Drop-in Mechanism

DoubleGraph is designed as a drop-in replacement. A Python user calls `cugraph.bfs(G, source=0)`, and through Cython bindings, it hits the underlying C++ library. At that precise moment, DoubleGraph hijacks the control flow.

The architecture is split into 4 strict layers:

```text
[ Python API ] -> [ Cython / pylibcugraph ] -> [ libcugraph C++ API ]
                                                      |
======================================================|================================
LAYER 4: INTEGRATION LAYER (cpp/src/aai/integration/) v
Intercepts the cuGraph C++ template via `#ifdef AAI_ROUTE_BFS` template specialization.
Synchronizes the cuGraph stream -> extracts compact graph -> calls Layer 3.
---------------------------------------------------------------------------------------
LAYER 3: API DISPATCH (cpp/include/cugraph/aai/api/)
Provides declarations for the 4-way dispatch (base, seg, mask, seg_mask).
---------------------------------------------------------------------------------------
LAYER 2: AAI INFRASTRUCTURE (cpp/include/cugraph/aai/)
Defines `compact_graph_t` (raw CSR/CSC arrays) and `CachePool` (thread-local GPU mem).
---------------------------------------------------------------------------------------
LAYER 1: AAI IMPLEMENTATION (cpp/src/aai/impl/{TARGET_GPU}/)
The actual `.cu` files containing the GPU-specific algorithms. 
Each file pairs with a `.cu.flags` sidecar for precise nvcc compiler arguments.
=======================================================================================
```

---

## 3. Layer 1: The Graph Abstraction (`compact_graph_t`)

To write fast kernels, we cannot pass cuGraph's `graph_view_t` into device code. It is bloated with multi-GPU logic. We extract the raw pointers into a C-style struct called `compact_graph_t`.

**Location:** `cpp/include/cugraph/aai/compact_graph.hpp`

```cpp
template <typename vertex_t, typename edge_t>
struct compact_graph_t {
    const edge_t* offsets;        // CSR/CSC row pointers
    const vertex_t* indices;      // CSR/CSC column indices
    vertex_t number_of_vertices;
    edge_t number_of_edges;
    bool is_symmetric;
    bool is_multigraph;
    bool is_csc;                  // Deduces storage format

    // Optional: Degree-based sorting segments (Host Memory)
    std::optional<std::vector<vertex_t>> segment_offsets;

    // Optional: Edge filtering bitmask (Device Memory)
    const uint32_t* edge_mask = nullptr;
};
```

### Critical Engineering Decisions here:
1.  **Flattening Compile-Time State:** cuGraph uses a compile-time boolean `is_storage_transposed` to determine CSR vs. CSC. We extract this via the factory method `from_graph_view<bool is_csc_>()` and store it as a *runtime* boolean (`is_csc`). This prevents us from having to template every single CUDA kernel on the storage format.
2.  **The Segments Array:** If `segment_offsets` has a value, it means the graph has been degree-sorted into exactly 4 bins: High degree ($\ge 1024$), Mid degree ($32-1023$), Low degree ($1-31$), and Isolated ($0$). This is vital for assigning warps vs. threads in traversal algorithms to prevent warp divergence.

---

## 4. Layer 2: Zero-Overhead Resource Management (`CachePool`)

Iterative graph workflows (like GNN data loading or dynamic graph analytics) call the same algorithms repeatedly. Calling `cudaMalloc` and `cudaFree` on every API call is a massive latency bottleneck.

**Location:** `cpp/include/cugraph/aai/cache_pool.hpp`

We use a **Thread-Local LRU Cache** to reuse GPU memory allocations transparently.

```cpp
// 1. Inherit from Cacheable to enable type-erasure
struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int64_t capacity = 0;

    // 2. Only reallocate if the new graph is larger than our current capacity
    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            capacity = num_vertices;
        }
    }
    ~Cache() override { /* cudaFree calls */ }
};

void bfs_kernel_launcher(const graph32_t& graph) {
    // 3. Unique, stable pointer tag identifies this algorithm's cache
    static int tag; 
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cache.ensure(graph.number_of_vertices);
    
    // Execute kernels using cache.frontier_a...
}
```

*Why this works:* It provides zero-overhead GPU memory pooling without requiring global state or locking. The LRU capacity is capped at 8, meaning if a user runs 9 different algorithms, the oldest cache is automatically evicted (its destructor fires, freeing the VRAM).

---

## 5. Layer 3: The 4-Way Dispatch Execution Matrix

To avoid warp divergence, we strictly prohibit `if (has_mask)` inside CUDA kernels. Instead, we branch at the host layer using a 4-way dispatch pattern. Every algorithm has up to four independent `.cu` files.

1.  `algo()` — Base CSR/CSC processing.
2.  `algo_seg()` — Uses `segment_offsets` for degree-aware block scheduling.
3.  `algo_mask()` — Inlines bitmask checking into the edge traversal loops.
4.  `algo_seg_mask()` — Degree-aware + Masking.

For complex algorithms like **PageRank**, this multiplies. PageRank dispatch must account for: Edge Mask $\times$ Personalization $\times$ Weights $\times$ Segments $\times$ Precision (Float/Double) = up to **32 different routing paths**.

---

## 6. Layer 4: A100-Specific Kernel Deep Dive

The SM80 (A100) implementation files are tuned around its **40MB L2 Cache** and **108 SMs**.

### 6.1 BFS: Dual-Frontier & Direction Optimization
*File: `impl/a100/traversal/bfs.cu`*

A100 uses **Direction-Optimizing BFS** with a queue-to-bitmap conversion.

*   **Top-Down (Sparse):** The frontier is small. We use a queue. The kernel assigns **one Warp (32 threads) per frontier vertex**. Threads iterate over the adjacency list.
    *   *Optimization:* We use `__ballot_sync` to perform warp-level atomic aggregations. Instead of 32 threads calling `atomicAdd` on the global frontier counter, we count the successful discoveries in the warp, thread 0 does a single `atomicAdd`, broadcasts the base index via `__shfl_sync`, and lanes write to their offsets.
*   **Bottom-Up (Dense):** The frontier is huge. We switch to a Bitmap. Because A100 has 40MB of L2, an entire $1M$ vertex bitmap ($128$ KB) fits easily in L2 cache. The kernel spawns one thread per unvisited vertex, checking if *any* of its neighbors are in the frontier bitmap. It breaks early on the first hit.
*   **A100 Thresholds:**
    *   `TD -> BU`: When frontier size $> N / 20$ (5% of graph).
    *   `BU -> TD`: When frontier size $< N / 200$ (0.5% of graph).

### 6.2 Louvain: 3-Tier Adaptive Dispatch & Shared-Memory Hashing
*File: `impl/a100/community/louvain_f32.cu`*

Louvain modularity optimization requires grouping a vertex's neighbor weights by their community IDs. This requires heavy hash-table lookups. Since graphs coarsen iteratively, the degree distribution changes wildly. We use a 3-tier dispatch:

1.  **Tier 1 ($N \le 200$):** Serial CPU-like execution on a single GPU thread. Avoids launch overhead for tiny coarsened graphs.
2.  **Tier 2 (Avg Degree < 8):** One thread per vertex. Uses a 64-entry hash table stored entirely in registers/local memory.
3.  **Tier 3 (Avg Degree $\ge$ 8):** One WARP per vertex. Uses a **128-entry Hash Table in Shared Memory**.
    *   *Mechanism:* `__shared__ int32_t s_ht_keys[WARPS_PER_BLOCK * 128]`. Threads in a warp collaboratively read neighbors and use `atomicCAS` with linear probing to insert community IDs into the shared memory hash table.

### 6.3 PageRank: Fused SpMV & Warp-Level Reduction
*File: `impl/a100/link_analysis/pagerank.cu`*

Rather than relying on generic CSR-vector multiplication, we fuse the PageRank formula ($PR_{new} = \alpha \times SpMV + Base$) directly into the SpMV traversal.

To check for convergence ($L_1$ Norm difference), we use warp-level reduction inside the same kernel to avoid a separate reduce pass:
```cuda
// Warp-level reduction
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    diff += __shfl_down_sync(0xffffffff, diff, offset);
}
// Block-level reduction via shared memory
if (lane == 0) shared_diff[warp_id] = diff;
__syncthreads();
// Final atomicAdd to global memory by thread 0
```

### 6.4 WCC: Parallel Union-Find & Host-Mapped Flags
*File: `impl/a100/components/weakly_connected_components.cu`*

Instead of standard label propagation (which converges slowly on long chains), A100 WCC uses **Parallel Union-Find**.
1.  **Hook Sample:** `wcc_hook_sample` looks at only the first $K=2$ edges of every vertex to quickly form base trees.
2.  **Compress:** Flattens trees via path-halving (`parent[v] = parent[parent[v]]`).
3.  **Hook Full:** Processes remaining edges.
4.  **Zero-Copy Convergence:** We allocate the convergence flag via `cudaHostAllocMapped`. The GPU writes directly to host memory via PCIe, saving an explicit `cudaMemcpy` synchronization step.

### 6.5 Triangle Count: DAG Orientation & Bilateral Intersection
*File: `impl/a100/community/triangle_count.cu`*

1. Computes vertex degrees.
2. Orients the graph into a DAG (Directed Acyclic Graph) pointing from High-Degree to Low-Degree to limit redundant counting.
3. Sorts adjacency lists using `cub::DeviceSegmentedSort`.
4. Performs **Bilateral Set Intersection**: For every edge $(u, v)$, intersects the sorted neighbor list of $u$ with the sorted list of $v$ using binary search, pre-fetching bounds via `__ldg()`.

---

## 7. Cross-Architecture Algorithmic Divergence (A100 vs L4 vs A10G)

The core thesis of WarpSpeed is that *compiling the same code for different GPUs is insufficient*. We generate entirely different algorithms.

### BFS Case Study: A100 vs. L4 vs. A10G

| Feature | A100 (SM80) | L4 (SM89) | A10G (SM86) |
| :--- | :--- | :--- | :--- |
| **L2 Cache Size** | 40 MB | 48 MB | **6 MB (Bottleneck)** |
| **SM Count** | 108 | 58 | 80 |
| **Switching Model** | Hard heuristics (N/20, N/200) | **Degree-Cost Model** via CUB BlockReduce | Aggressive Const (ALPHA=4) |
| **Batching** | Single Level | Single Level | **2-Level Batching** |
| **State Polling** | `cudaMemcpy` | `cudaMemcpy` | **Pinned Host Memory** |
| **BU $\to$ TD Rebuild**| `bitmap_to_queue` | `bitmap_to_queue` | **Rebuild from distances array** |

**Deep Dive: Why is A10G so weird?**
The A10G has only 6MB of L2 Cache. In BFS, expanding a tiny frontier triggers a kernel launch, reads memory, writes memory, and syncs with the CPU. On the A10G, this PCIe/Launch overhead dominates the actual compute. 

To solve this, A10G uses **2-Level Batching**. 
If the frontier is an appropriate size, the A10G implementation executes `depth + 1` and `depth + 2` in a single CPU sequence. The second kernel (`bfs_topdown_devsize_kernel`) reads the frontier size *directly from device memory*, completely bypassing the CPU sync. We then use `cudaHostAlloc` pinned memory to asynchronously poll the final frontier size, halving the roundtrip latency.

Furthermore, because A10G's L2 is so small, maintaining the `frontier_queue` during Bottom-Up mode thrashes the cache. We completely drop the queue. When it's time to switch back to Top-Down, a special kernel rebuilds the queue by scanning the `distances` array.

---

## 8. The Integration Layer & Build Pipeline

### Template Specialization Routing (`cpp/src/aai/integration/`)
We override cuGraph's default algorithms using template specialization triggered by CMake definitions.

```cpp
#ifdef AAI_ROUTE_BFS

template <>
void bfs<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    // ... args
) {
    auto compact_graph = aai::graph32_t::from_graph_view(graph_view);
    
    // CRITICAL: Must sync cuGraph's stream because AAI uses Stream 0
    handle.sync_stream();

    // 4-way dispatch
    if (compact_graph.edge_mask) aai::bfs_mask(...);
    else aai::bfs(...);

    // CRITICAL: Sync device and catch errors before returning to cuGraph
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) CUGRAPH_FAIL(...);
}
#endif
```

### The CMake Build Pipeline (`cpp/CMakeLists.txt`)

1.  **Target Injection:** `-DTARGET_GPU=A100` tells CMake to set `AAI_TARGET_GPU_SUFFIX="a100"`.
2.  **Globbing:** CMake globs only `src/aai/impl/a100/**/*.cu`. This prevents symbol collisions.
3.  **Sidecar Flags:** For every `.cu` file found, CMake checks for a `<filename>.cu.flags` sidecar file. 
    *   If it finds `--use_fast_math`, it applies it specifically to that source file using `set_source_files_properties`.
    *   If it finds `--maxrregcount=48`, it restricts registers for that specific kernel to increase occupancy.
4.  **RDC Sub-Library:** If a flags file contains `--rdc=true` (required for cooperative kernels with grid-wide `__syncthreads`, like `bfs_mask`), CMake separates those files into a static library `cugraph_aai_rdc` compiled with `CUDA_SEPARABLE_COMPILATION ON`.
5.  **Routing:** It sets `-DAAI_ROUTE_BFS` on `integration/traversal/bfs.cu` to activate the specialization.

---

## 9. Hard Constraints & Known Limitations

When modifying or running DoubleGraph, the following physical limits apply:

1.  **Single GPU Only:** `number_of_local_edge_partitions() != 1` will throw a runtime error. Distributed multi-GPU graphs are not supported by AAI kernels.
2.  **Int32 Vertex/Edge Types Only:** Only `int32_t` templates are specialized. Graphs exceeding 2.1 billion edges will fall back to standard cuGraph.
3.  **No Hypersparse (DCS):** `compact_graph_t` demands exactly 5 segment offsets (4 degree bins). Hypersparse graphs break this assumption.
4.  **Stream 0 Execution:** AAI executes on the default stream. `handle.sync_stream()` is mandatory before execution, preventing async overlap with other cuGraph tasks.
5.  **No Hot-Path Error Checking:** To maximize speed, `cudaMalloc` and `cudaMemcpy` inside implementations do not use `CUDA_TRY`. Errors are caught at the integration boundary via `cudaGetLastError()`. Use `CUDA_LAUNCH_BLOCKING=1` to debug.

---

## 10. Porting to Future Hardware: H100 (SM90) and B200 (SM100+)

Adapting DoubleGraph for Hopper and Blackwell architectures requires exploiting new hardware primitives.

### H100 (Hopper - SM90)
*   **Tensor Memory Accelerator (TMA):** H100 allows async, hardware-driven bulk memory copies. Instead of threads issuing loads for neighbor lists in BFS, use TMA to fetch adjacency lists asynchronously, freeing the SM to process distances.
*   **Thread Block Clusters & Distributed Shared Memory (DSmem):** Louvain's Tier 3 currently limits its shared memory hash table to 128 entries per warp. By grouping blocks into clusters, you can stripe a massive hash table across multiple SMs using DSmem.
*   **Warp Specialization:** For heavy traversals, dedicate one warp specifically to fetching `offsets` and `indices` (Producer), and another warp to evaluating distances and atomic updates (Consumer).

### B200 / B300 (Blackwell - SM100+)
*   **Warp Size Risks:** If Blackwell fundamentally alters the hardware warp size (e.g., away from 32), all hardcoded `threadIdx.x & 31`, `__ballot_sync(0xffffffff)`, and `__shfl_down_sync(..., 16)` operations in AAI will break. These must be abstracted into macros.
*   **HBM3e Bandwidth:** Algorithms currently designated as memory-bound (like Bottom-Up BFS) will execute substantially faster. The switching thresholds (e.g., `N/20` or `alpha` cost models) must be recalibrated, as traversing the whole graph becomes drastically cheaper relative to atomic-heavy queue operations.

---

## 11. Transferable Engineering Strategies

For systems engineers looking to adapt this approach to other domains (e.g., PyTorch kernels, custom LLM inference engines):

1.  **Compile-Time GPU Dispatch:** Shipping multiple targeted wheels (one per GPU) completely eliminates `if (is_a100)` branching in hot C++ code, improving instruction cache hit rates.
2.  **The `.cu.flags` Sidecar Pattern:** Avoid polluting global `CMakeLists.txt` with compiler flags. Sidecars allow extreme micro-optimization (like setting `--maxrregcount=64` just for `sorensen_all_pairs.cu`).
3.  **Template Hijacking:** You can rewrite the entire backend of a massive library without modifying its Python or Cython layers by precisely specializing its C++ templates and manipulating include guards.
4.  **Static Tag LRU Caching:** Using `static int tag;` as a pointer reference into an LRU `CachePool` is a thread-safe, string-free, zero-overhead way to manage dynamic GPU resource lifetimes across disjoint API calls.