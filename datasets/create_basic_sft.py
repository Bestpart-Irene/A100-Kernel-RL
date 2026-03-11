#!/usr/bin/env python3
"""
Create basic SFT dataset from existing kernels and simple CUDA examples.

This focuses on teaching the model fundamental CUDA patterns before
moving to complex graph algorithms.
"""

import json
import os
from pathlib import Path

# A100-specific context
A100_CONTEXT = """
GPU: A100 (sm_80), 40MB L2 cache, 108 SMs
Architecture: Ampere, Tensor Cores available  
Memory: HBM2e, bandwidth 1.55 TB/s

A100 Optimization Guidelines:
- Exploit 40MB L2 for bitmaps < 128KB
- Use degree-aware scheduling for >1M vertices
- Consider warp-per-vertex for high-degree nodes
- Use __shfl_sync for warp-level reductions
- Leverage cooperative groups for complex synchronization

Requirements:
- Output ONLY CUDA code, no explanations
- Include extern "C" __global__ signature
- Keep kernel under 800 tokens
"""

def create_wcc_sft():
    """Create WCC SFT entry from baseline_wcc.cu"""
    kernel_path = Path(__file__).parent.parent / "kernels/baseline_wcc.cu"
    
    with open(kernel_path, 'r') as f:
        content = f.read()
    
    # Extract the main kernels
    lines = content.split('\n')
    kernels = []
    current_kernel = []
    in_kernel = False
    
    for line in lines:
        if '__global__' in line:
            if current_kernel:
                kernels.append('\n'.join(current_kernel))
            current_kernel = [line]
            in_kernel = True
        elif in_kernel and line.strip() == '}':
            current_kernel.append(line)
            kernels.append('\n'.join(current_kernel))
            current_kernel = []
            in_kernel = False
        elif in_kernel:
            current_kernel.append(line)
    
    # Create SFT entries
    sft_entries = []
    
    # WCC Initialize
    init_kernel = None
    hook_kernel = None
    
    for kernel in kernels:
        if 'wcc_initialize' in kernel:
            init_kernel = kernel
        elif 'wcc_hook_phase' in kernel:
            hook_kernel = kernel
    
    if init_kernel and hook_kernel:
        # Combine into one entry
        combined = f"""{init_kernel}

{hook_kernel}"""
        
        sft_entries.append({
            "prompt": f"""Write CUDA kernels for Weakly Connected Components using Union-Find algorithm on A100.

{A100_CONTEXT}

Requirements:
- Implement parallel union-find with path compression
- Use atomic operations for thread safety
- Include initialization and hook phase kernels
- Process CSR graph format (row_ptr, col_idx)""",
            
            "completion": combined,
            "kernel_id": "a100/components/wcc_baseline",
            "category": "components",
            "algorithm": "weakly_connected_components",
            "variant": "base",
            "curriculum_level": 0
        })
    
    return sft_entries

def create_vector_ops_sft():
    """Create basic vector operation SFT entries."""
    
    sft_entries = []
    
    # Vector Add
    vector_add = '''extern "C" __global__ void vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}'''
    
    sft_entries.append({
        "prompt": f"""Write a vector addition CUDA kernel for A100.

{A100_CONTEXT}

Requirements:
- Add two vectors element-wise
- Use grid-stride loop for large arrays
- Optimize for memory bandwidth""",
        
        "completion": vector_add,
        "kernel_id": "a100/ops/vector_add",
        "category": "ops",
        "algorithm": "vector_add",
        "variant": "base",
        "curriculum_level": 0
    })
    
    # Matrix Multiply (simplified)
    matmul = '''extern "C" __global__ void matmul_simple(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}'''
    
    sft_entries.append({
        "prompt": f"""Write a simple matrix multiplication CUDA kernel for A100.

{A100_CONTEXT}

Requirements:
- Compute C = A * B where A is MxK, B is KxN, C is MxN
- Use 2D thread blocks
- No shared memory optimization (keep it simple)""",
        
        "completion": matmul,
        "kernel_id": "a100/ops/matmul_simple",
        "category": "ops",
        "algorithm": "matrix_multiply",
        "variant": "base",
        "curriculum_level": 1
    })
    
    return sft_entries

def create_graph_traversal_sft():
    """Create graph traversal SFT entries."""
    
    sft_entries = []
    
    # BFS
    bfs = '''extern "C" __global__ void bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ frontier,
    int* __restrict__ distances,
    int* __restrict__ next_frontier,
    int* __restrict__ next_size,
    int source,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes one vertex
    if (tid < *next_size) {
        int v = next_frontier[tid];
        
        // Explore neighbors
        for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
            int u = col_idx[e];
            
            // Try to claim this vertex
            if (atomicCAS(&distances[u], -1, level + 1) == -1) {
                // Successfully discovered
                int idx = atomicAdd(next_size, 1);
                frontier[idx] = u;
            }
        }
    }
}'''
    
    sft_entries.append({
        "prompt": f"""Write a BFS kernel for A100 using frontier-based approach.

{A100_CONTEXT}

Requirements:
- Use CSR graph format
- Implement frontier expansion
- Use atomic operations for deduplication
- Track distances from source""",
        
        "completion": bfs,
        "kernel_id": "a100/traversal/bfs_frontier",
        "category": "traversal",
        "algorithm": "bfs",
        "variant": "base",
        "curriculum_level": 1
    })
    
    return sft_entries

def create_reduction_sft():
    """Create parallel reduction SFT entries."""
    
    sft_entries = []
    
    # Warp-level reduction
    reduction = '''extern "C" __global__ void warp_reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int tid = threadIdx.x;
    int lane = tid & 31;  // Lane ID within warp
    int wid = tid >> 5;  // Warp ID
    
    // Load data into shared memory
    extern __shared__ float sdata[];
    sdata[tid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Warp-level reduction using shuffle
    float val = sdata[tid];
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // First thread in warp writes result
    if (lane == 0) {
        output[wid] = val;
    }
}'''
    
    sft_entries.append({
        "prompt": f"""Write a parallel reduction kernel using warp-level shuffle instructions for A100.

{A100_CONTEXT}

Requirements:
- Use __shfl_down_sync for warp reduction
- Minimize shared memory usage
- Handle arbitrary array sizes""",
        
        "completion": reduction,
        "kernel_id": "a100/ops/warp_reduce_sum",
        "category": "ops",
        "algorithm": "reduction",
        "variant": "base",
        "curriculum_level": 1
    })
    
    return sft_entries

def main():
    """Create basic SFT dataset."""
    print("Creating basic SFT dataset...")
    
    sft_entries = []
    
    # Add different categories
    sft_entries.extend(create_wcc_sft())
    sft_entries.extend(create_vector_ops_sft())
    sft_entries.extend(create_graph_traversal_sft())
    sft_entries.extend(create_reduction_sft())
    
    # Write dataset
    output_path = Path(__file__).parent / "basic_cuda_sft.jsonl"
    
    print(f"Writing {len(sft_entries)} entries to {output_path}")
    
    with open(output_path, 'w') as f:
        for entry in sft_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Statistics
    print("\n=== Basic SFT Dataset Statistics ===")
    print(f"Total entries: {len(sft_entries)}")
    
    level_counts = {}
    category_counts = {}
    
    for entry in sft_entries:
        level = entry['curriculum_level']
        category = entry['category']
        
        level_counts[level] = level_counts.get(level, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\nCurriculum levels:")
    for level in sorted(level_counts):
        print(f"  Level {level}: {level_counts[level]} entries")
    
    print("\nCategories:")
    for cat in sorted(category_counts):
        print(f"  {cat}: {category_counts[cat]} entries")
    
    # Show sample
    print("\n=== Sample Entry ===")
    sample = sft_entries[0]
    print(f"Kernel ID: {sample['kernel_id']}")
    print(f"Prompt length: {len(sample['prompt'])} chars")
    print(f"Completion length: {len(sample['completion'])} chars")

if __name__ == "__main__":
    main()
