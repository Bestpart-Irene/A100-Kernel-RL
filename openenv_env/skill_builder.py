"""Dynamic SKILL.md generation per target GPU architecture.

Extracted from the Implementation Spec Section 1.3 / Section 4.2.
Each GPU gets architecture-specific optimization guidance.
Includes real A100 patterns from doubleGraph production kernels (192 files, 84K+ lines).
"""
from __future__ import annotations

import os

from openenv_env.gpu_registry import get_gpu_spec


def build_skill_md(gpu_name: str = "a100") -> str:
    """Generate GPU-specific SKILL.md content.

    First tries to load a static skill file (e.g. skill_a100.md).
    Falls back to dynamic generation from GPU_REGISTRY specs.
    For A100, appends real expert patterns from doubleGraph production kernels.
    """
    # Try static file first (user may have customized it)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_file = os.getenv("KERNELFORGE_SKILL_FILE", "")
    if env_file:
        path = os.path.join(root, env_file)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                skill = f.read()
            if gpu_name.lower() == "a100":
                skill = _append_a100_patterns(skill)
            return skill

    static_path = os.path.join(root, f"skill_{gpu_name.lower()}.md")
    if os.path.exists(static_path):
        with open(static_path, encoding="utf-8") as f:
            skill = f.read()
        if gpu_name.lower() == "a100":
            skill = _append_a100_patterns(skill)
        return skill

    # Dynamic generation from registry
    skill = _generate_skill_md(gpu_name)
    if gpu_name.lower() == "a100":
        skill = _append_a100_patterns(skill)
    return skill


def _generate_skill_md(gpu_name: str) -> str:
    """Generate SKILL.md dynamically from GPU_REGISTRY specs."""
    spec = get_gpu_spec(gpu_name)
    name = spec["name"]

    skill = f"""# SKILL.md — {name}-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA {name} ({spec['arch']})
- SMs: {spec['sms']} | L2 Cache: {spec['l2_cache_mb']} MB
- Shared Memory/SM: {spec['smem_per_sm_kb']} KB
- HBM Bandwidth: {spec['hbm_bandwidth_tbs']} TB/s
- Registers/SM: {spec['registers_per_sm']}
- nvcc: {spec.get('nvcc_flag', '-arch=' + spec['arch'])} -O3

## Per-Kernel Compilation Flags
Specify in a comment: // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed: --use_fast_math, --maxrregcount=N (16-128),
         --rdc=true, --extra-device-vectorization

## Optimization Priorities

### Priority 1: Algorithmic Reduction (>50% impact)
- Algebraic simplification: reduce complexity before optimizing.
  Example: diag(A) @ B = row-wise scaling. O(N^2*M) -> O(N*M).
- Kernel fusion: merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization.
- Operator rearrangement: restructure computation order.
"""

    # Architecture-specific memory guidance
    skill += "\n### Priority 2: Memory Hierarchy (20-50% impact)\n"

    if gpu_name.lower() == "a100":
        skill += f"""- L2 cache pinning: {name} has {spec['l2_cache_mb']}MB L2. Pin frequently accessed arrays:
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
- Vectorized loads: float4 for 16-byte aligned access (4x transaction efficiency)
- Shared memory tiling: pad arrays float tile[32][33] to avoid bank conflicts
- Memory coalescing: consecutive threads access consecutive addresses
- __ldg() for read-only data via texture cache path
"""
    elif gpu_name.lower() in {"h100", "h200"}:
        skill += f"""- TMA (Tensor Memory Accelerator): bulk async copy from global to shared.
  One thread initiates copy, 127 freed for computation. 1.45 TB/s.
- Distributed shared memory: SMs in a cluster access each other's SMEM.
- L2 cache pinning: {name} has {spec['l2_cache_mb']}MB L2 (larger than A100).
- Thread Block Clusters: __cluster_dims__ for cross-block shared memory.
- Vectorized loads: float4 for aligned accesses.
"""
    elif gpu_name.lower() == "b200":
        skill += f"""- HBM3e: ~{spec['hbm_bandwidth_tbs']} TB/s bandwidth.
- NVLink 5.0: 1.8 TB/s bidirectional for multi-GPU.
- L2 cache: ~{spec['l2_cache_mb']}MB. Aggressive caching strategies viable.
- TMA + Distributed SMEM from Hopper still apply.
- Profile before assuming A100/H100 patterns transfer.
"""
    else:
        skill += f"- L2 cache: {spec['l2_cache_mb']}MB. Profile to determine best strategy.\n"

    skill += """
### Priority 3: Compute Optimization (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp communication.
- Occupancy tuning: __launch_bounds__(threads, blocks) for register control.
- TF32 for matmul: leverages Tensor Cores for ~3x GEMM throughput.
"""

    if spec.get("has_tma"):
        skill += "- Warp specialization: persistent producer/consumer warp roles.\n"

    skill += """
### Priority 4: Library Integration
- cuBLAS for GEMM: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
- cuDNN for Conv: cudnnConvolutionBiasActivationForward fuses conv+bias+activation
- cuSPARSE for SpMV: cusparseSpMV with CSR format

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input shape
"""
    return skill


def _append_a100_patterns(skill_md: str) -> str:
    """Append real A100 expert patterns extracted from doubleGraph production kernels.

    These patterns come from 192 A100-optimized CUDA kernels (84K+ lines)
    achieving 3.6x average speedup over cuGraph on graph algorithms.
    """
    patterns = """

## A100 Expert Patterns (from doubleGraph — 3.6x avg speedup over cuGraph)

### Pattern 1: Degree-Based Kernel Dispatch
For graph algorithms, choose dispatch strategy based on vertex degree:
- **degree >= 8**: warp-cooperative (1 warp per vertex, shared-mem hash table)
- **degree < 8**: thread-per-vertex (register-local accumulation)
- **n <= 200**: serial (single thread, avoids launch overhead for tiny graphs)

```cuda
// Warp-cooperative: each warp handles one vertex
int warp_id = threadIdx.x / 32;
int lane = threadIdx.x % 32;
int v = (blockIdx.x * WARPS_PER_BLOCK) + warp_id;
// Edges processed cooperatively: lane stride
for (int e = start + lane; e < end; e += 32) { ... }
```
Source: louvain_f32.cu lines 95-175

### Pattern 2: Warp-Level Shared-Memory Hash Tables
Use per-warp shared-memory hash tables (128 entries) for accumulating
per-community edge weights without global memory pressure:

```cuda
__shared__ int32_t s_ht_keys[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];
__shared__ float   s_ht_vals[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];
// Linear probing with atomicCAS for thread-safe insertion
uint32_t h = (uint32_t)(key * 2654435761u) % HT_SIZE_PER_WARP;
int32_t old = atomicCAS(&my_keys[slot], -1, key);
if (old == -1 || old == key) { atomicAdd(&my_vals[slot], w); }
__syncwarp();  // Lane 0 reads final result
```
Source: louvain_f32.cu lines 111-162

### Pattern 3: Zero-Copy Convergence Flag
For iterative algorithms (WCC, PageRank, eigenvector), avoid cudaMemcpy
round-trip per iteration. Use pinned memory visible to both host and device:

```cuda
struct Cache : Cacheable {
    int* d_flag = nullptr;
    int* h_flag = nullptr;
    Cache() {
        cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_flag, h_flag, 0);
    }
};
// In kernel: atomicOr(d_flag, 1) to signal convergence change
// On host: read *h_flag directly (zero-copy), no cudaMemcpy needed
```
Source: weakly_connected_components.cu lines 24-40

### Pattern 4: Bitmap Frontier for BFS / Direction-Optimizing Traversal
For power-law graphs, switch between top-down (queue) and bottom-up (bitmap):

```cuda
__device__ __forceinline__ bool bmp_test(const uint32_t* bmp, int v) {
    return (bmp[v >> 5] >> (v & 31)) & 1u;
}
__device__ __forceinline__ void bmp_set_atomic(uint32_t* bmp, int v) {
    atomicOr(&bmp[v >> 5], 1u << (v & 31));
}
// Direction switch: frontier_size < n/alpha → top-down, else bottom-up
```
Source: bfs_direction_optimizing.cu lines 72-78

### Pattern 5: Path-Halving Union-Find (Lock-Free)
For WCC / connected components, non-atomic path compression:

```cuda
__device__ __forceinline__ int find_root(int* parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        p = parent[parent[p]];  // Path halving — no atomics needed
    }
    return p;
}
```
Source: weakly_connected_components.cu line 48

### Pattern 6: Compilation Flag Tuning by Algorithm Class
Match nvcc flags to algorithm characteristics:

| Algorithm Class | Flag | Why |
|----------------|------|-----|
| Latency-bound (eigenvector, leiden) | `--maxrregcount=40-48` | Force high occupancy to hide memory latency |
| Compute-bound (link prediction all-pairs) | `--maxrregcount=56-64` | Allow more registers per thread for throughput |
| Memory-bound (WCC, ratio cut) | `--extra-device-vectorization` | Auto-vectorize memory access patterns |
| Temporal locality (ego graph extraction) | `-Xptxas -dlcm=ca` | Cache ALL loads in L1 |
| Separate compilation (BFS mask, core number) | `--rdc=true` | Relocatable device code |
| ALL kernels | `--use_fast_math` | Trade precision for throughput (acceptable for graph algorithms) |

Specify flags via: `// CU_FLAGS: --use_fast_math --maxrregcount=48`

### Pattern 7: __launch_bounds__ for Register Control
Explicitly set launch bounds to let the compiler optimize register allocation:

```cuda
__global__ void __launch_bounds__(256)     // 256 threads, compiler picks min blocks
wcc_init(int* parent, int n) { ... }

__global__ void __launch_bounds__(BLOCK_SIZE, 6)  // 6 blocks/SM minimum
local_move_warp(...) { ... }
```
The second parameter (min blocks) forces lower register usage for higher occupancy.
"""
    return skill_md + patterns
