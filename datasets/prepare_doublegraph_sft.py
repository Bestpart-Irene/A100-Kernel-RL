#!/usr/bin/env python3
"""
Prepare DoubleGraph A100 kernels for SFT training.

Extracts kernels from the DoubleGraph dataset and formats them as
prompt-completion pairs for supervised fine-tuning.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

# A100-specific context for prompts
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

def algorithm_description(category: str, algorithm: str, variant: str) -> str:
    """Generate human-readable description for the algorithm."""
    descriptions = {
        "centrality": {
            "betweenness_centrality": "Compute betweenness centrality using Brandes' algorithm",
            "edge_betweenness_centrality": "Compute edge betweenness centrality",
            "eigenvector_centrality": "Compute eigenvector centrality using power iteration",
            "katz_centrality": "Compute Katz centrality with damping factor",
            "pagerank": "Compute PageRank using power iteration",
            "hits": "Compute HITS algorithm (hubs and authorities)"
        },
        "community": {
            "louvain": "Detect communities using Louvain modularity optimization",
            "triangle_count": "Count triangles in the graph"
        },
        "components": {
            "weakly_connected_components": "Find weakly connected components using union-find"
        },
        "traversal": {
            "bfs": "Breadth-First Search from source vertex",
            "sssp": "Single-Source Shortest Paths using Dijkstra's algorithm"
        },
        "link_analysis": {
            "pagerank": "PageRank algorithm with damping",
            "hits": "HITS algorithm implementation"
        },
        "link_prediction": {
            "jaccard_all_pairs": "Jaccard similarity for all node pairs",
            "cosine": "Cosine similarity for node pairs",
            "overlap": "Overlap coefficient similarity"
        }
    }
    
    base_desc = descriptions.get(category, {}).get(algorithm, f"Implement {algorithm}")
    
    # Add variant info
    variant_info = {
        "base": "",
        "seg": " with degree-aware segmentation",
        "mask": " with edge mask support",
        "seg_mask": " with both segmentation and edge masking"
    }
    
    return base_desc + variant_info.get(variant, "")

def extract_kernel_source(kernel_info: Dict) -> str:
    """Extract the main kernel function from a .cu file."""
    # Update path to point to cloned doublegraph repository
    root_dir = Path(__file__).parent.parent
    source_path = root_dir / "doublegraph_src" / kernel_info.get('source_path', '')
    
    if not source_path.exists():
        return f"// ERROR: Source file not found: {source_path}\n"
    
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Extract the main kernel function (extern "C" __global__)
    lines = content.split('\n')
    kernel_start = -1
    brace_count = 0
    kernel_lines = []
    
    for i, line in enumerate(lines):
        if 'extern "C"' in line and '__global__' in line:
            kernel_start = i
            # Add includes and helper functions before kernel
            # Find the last #include or empty line before kernel
            j = i - 1
            while j >= 0 and (lines[j].strip().startswith('#') or 
                             lines[j].strip().startswith('using') or
                             lines[j].strip().startswith('template') or
                             not lines[j].strip()):
                j -= 1
            j += 2  # Include the line after the last non-include
            
            # Add essential includes
            kernel_lines = [
                "#include <cuda_runtime.h>",
                "#include <cstdint>",
                "#include <device_functions.h>",
                ""
            ]
            
            # Add any custom includes from original
            for line in lines[j:i]:
                if line.strip().startswith('#include'):
                    kernel_lines.append(line)
            
            kernel_lines.extend(lines[i:])
            break
    
    if kernel_start == -1:
        return f"// ERROR: No extern \"C\" __global__ function found in {source_path}\n"
    
    # Extract just the kernel function by tracking braces
    in_kernel = False
    kernel_only = []
    
    for line in kernel_lines:
        if 'extern "C"' in line and '__global__' in line:
            in_kernel = True
        
        if in_kernel:
            kernel_only.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0 and '}' in line:
                break
    
    return '\n'.join(kernel_only)

def create_sft_entry(kernel_info: Dict) -> Dict:
    """Create an SFT entry from kernel manifest data."""
    kernel_id = kernel_info['kernel_id']
    category = kernel_info['category']
    algorithm = kernel_info['algorithm_name']
    variant = kernel_info['variant']
    
    # Build prompt
    task_desc = algorithm_description(category, algorithm, variant)
    
    prompt = f"""Write a CUDA kernel for A100 (sm_80) to {task_desc}.

{A100_CONTEXT}

Algorithm: {algorithm}
Category: {category}
Variant: {variant}

Output format:
extern "C" __global__ void kernel_name(...) {{
    // Implementation
}}"""

    # Extract kernel source using the updated function
    completion = extract_kernel_source(kernel_info)
    
    return {
        "prompt": prompt,
        "completion": completion,
        "kernel_id": kernel_id,
        "category": category,
        "algorithm": algorithm,
        "variant": variant,
        "metadata": {
            "line_count": kernel_info.get('line_count', 0),
            "uses_cooperative_groups": kernel_info.get('uses_cooperative_groups', False),
            "uses_atomic_ops": kernel_info.get('uses_atomic_ops', False),
            "requires_rdc": kernel_info.get('requires_rdc', False),
            "flags": kernel_info.get('flags', [])
        }
    }

def main():
    """Main function to prepare SFT dataset."""
    # Paths
    root_dir = Path(__file__).parent.parent
    manifest_path = root_dir / "docs/research/doublegraph/doublegraph_a100_manifest.jsonl"
    output_path = root_dir / "datasets/doublegraph_sft_from_manifest.jsonl"
    
    print(f"Loading DoubleGraph manifest from {manifest_path}")
    
    # Load manifest
    kernels = []
    with open(manifest_path, 'r') as f:
        for line in f:
            kernels.append(json.loads(line))
    
    print(f"Found {len(kernels)} kernels")
    
    # Create SFT entries
    sft_entries = []
    curriculum_levels = {
        "components": 0,  # Simple
        "traversal": 0,  # Simple  
        "centrality": 1,  # Medium
        "community": 1,   # Medium
        "link_analysis": 1,  # Medium
        "link_prediction": 2,  # Complex
    }
    
    for kernel in kernels:
        entry = create_sft_entry(kernel)
        entry['curriculum_level'] = curriculum_levels.get(kernel['category'], 1)
        sft_entries.append(entry)
    
    # Write SFT dataset
    print(f"Writing {len(sft_entries)} SFT entries to {output_path}")
    
    with open(output_path, 'w') as f:
        for entry in sft_entries:
            f.write(json.dumps(entry) + '\n')
    
    # Statistics
    print("\n=== SFT Dataset Statistics ===")
    print(f"Total entries: {len(sft_entries)}")
    
    level_counts = {}
    for entry in sft_entries:
        level = entry['curriculum_level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("\nCurriculum distribution:")
    for level in sorted(level_counts):
        print(f"  Level {level}: {level_counts[level]} entries")
    
    category_counts = {}
    for entry in sft_entries:
        cat = entry['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat in sorted(category_counts):
        print(f"  {cat}: {category_counts[cat]} entries")
    
    # Sample entry
    print("\n=== Sample Entry ===")
    sample = sft_entries[0]
    print(f"Kernel ID: {sample['kernel_id']}")
    print(f"Prompt (first 200 chars): {sample['prompt'][:200]}...")
    print(f"Completion (first 200 chars): {sample['completion'][:200]}...")

if __name__ == "__main__":
    main()
