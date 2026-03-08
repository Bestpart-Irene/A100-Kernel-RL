# SKILL.md — A100-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA A100 (Ampere, sm_80)
- SMs: 108 | L2 Cache: 40 MB | Shared Memory/SM: 164 KB
- HBM2e: 2.0 TB/s | Registers/SM: 65,536
- nvcc: -arch=sm_80 -O3

## Your Task
Write an optimized CUDA C++ extension that accelerates the given
PyTorch operator. Your code runs on A100.

## Kernel Interface
Your code MUST define:
    extern "C" void optimized_forward(/* problem-specific args */);

## Per-Kernel Compilation Flags
You may specify compilation flags in a comment:
    // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed: --use_fast_math, --maxrregcount=N (16-128),
         --rdc=true, --extra-device-vectorization

## A100 Optimization Techniques (Priority Order)

### Priority 1: Algorithmic Reduction (>50% impact)
- Algebraic simplification: reduce computation before optimizing.
  Example: diag(A) x B = row-wise scaling (O(N^2 M) -> O(NM), 73x speedup)
- Kernel fusion: merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization and kernel launch overhead.
- Operator rearrangement: restructure computation order.
  Example: x @ (sum_j w_j^T) / 2 instead of sum_j (x @ w_j^T / 2) (24x speedup)

### Priority 2: A100 Memory Hierarchy (20-50% impact)
- L2 cache pinning: A100 has 40MB L2. Pin frequently accessed data:
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
    // Then set stream attribute with cudaAccessPropertyPersisting
  For arrays up to 10M elements (40MB), entire working set fits in L2.
  Drops random access from ~200 cycles (HBM) to ~30-40 cycles (L2).
- Vectorized loads: float4 for 16-byte aligned access (4x transaction efficiency)
- Memory coalescing: consecutive threads access consecutive addresses
- Shared memory tiling: pad arrays float tile[32][33] to avoid 32-bank conflicts
  A100 has 164KB SMEM per SM — use it aggressively.
- __ldg() for read-only data via texture cache path
- cp.async pipelining: cuda::memcpy_async for global->shared (sm_80 exclusive)

### Priority 3: A100 Compute (10-20% impact)
- Warp primitives: __shfl_sync, __ballot_sync for intra-warp ops.
  Hand-rolled reductions give tighter register control than CUB library:
    __device__ float warp_reduce_sum(float val) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }
- Occupancy tuning: __launch_bounds__(256, 4) for register control.
  Use cudaOccupancyMaxActiveBlocksPerMultiprocessor.
- Cooperative kernels: for iterative algorithms, use
  cg::this_grid().sync() to avoid kernel launch overhead.
  Requires: // CU_FLAGS: --rdc=true
- TF32 for matmul: ~3x GEMM throughput on A100 Tensor Cores.
  Use cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32.

### Priority 4: Library Integration
- cuBLAS for GEMM: cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
- cuDNN for Conv: cudnnConvolutionBiasActivationForward (fused conv+bias+activation)
- cuSPARSE for SpMV: cusparseSpMV with CSR format

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT use Triton or any kernel generation framework
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input shape
