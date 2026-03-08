"""
Extract doubleGraph A100 production kernels into TRL-compatible training datasets.

Reads 192 .cu files from doubleGraph/cpp/src/aai/impl/a100/, pairs them with
topology-aware prompts and compilation flags, outputs:
  - doublegraph_sft.jsonl           (HF messages format for SFT/RFT)

Note: Legacy outputs (doublegraph_a100_kernels.jsonl, doublegraph_grpo_prompts.jsonl)
archived to archive/datasets_legacy/. Use build_combined_dataset.py for RL training.

Usage:
    python datasets/extract_doublegraph_a100.py [--doublegraph-root PATH]
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────────

DOUBLEGRAPH_ROOT = os.getenv(
    "DOUBLEGRAPH_ROOT",
    str(Path(__file__).resolve().parents[1].parent / "doubleGraph"),
)
A100_IMPL_DIR = os.path.join(DOUBLEGRAPH_ROOT, "cpp", "src", "aai", "impl", "a100")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_GPU = "A100"
TARGET_ARCH = "sm_80"

# ── Topology → Algorithm mapping ──────────────────────────────────────────

CATEGORY_META: dict[str, dict[str, Any]] = {
    "traversal": {
        "topology": "power-law",
        "topology_desc": (
            "power-law web graph (HiBench-class, skewed degree distribution with hub vertices). "
            "Standard per-vertex parallelism causes catastrophic warp divergence on hubs."
        ),
        "pattern": "Direction-optimizing traversal, bitmap frontier, __ballot_sync warp voting",
        "datasets": ["hibench_small", "cyber.csv"],
    },
    "components": {
        "topology": "sparse-islands",
        "topology_desc": (
            "sparse graph with disconnected islands (netscience-class, 1,461 vertices, many components). "
            "Naive atomic Union-Find causes contention hotspots on root vertices."
        ),
        "pattern": "Non-atomic Union-Find, path halving, zero-copy convergence flag, __launch_bounds__(256)",
        "datasets": ["netscience.csv", "karate.csv"],
    },
    "link_analysis": {
        "topology": "dense-regular",
        "topology_desc": (
            "dense hierarchical network (email-Eu-core-class, 25K edges). "
            "Iterative convergence algorithm dominated by SpMV memory bandwidth."
        ),
        "pattern": "SpMV + sum-reduce pipeline, pinned memory convergence check, cudaHostAlloc zero-copy flag",
        "datasets": ["email-Eu-core.csv", "polbooks.csv"],
    },
    "community": {
        "topology": "dense-community",
        "topology_desc": (
            "social network with distinct community clustering (Karate/Dolphins-class). "
            "Modularity optimization requires tracking per-community edge weight sums efficiently."
        ),
        "pattern": "Warp-level shared-memory hash tables (128 entries/warp), degree-based dispatch (warp/thread/serial)",
        "datasets": ["karate.csv", "dolphins.csv", "polbooks.csv"],
    },
    "centrality": {
        "topology": "dense-regular",
        "topology_desc": (
            "dense graph requiring multi-source BFS or power iteration (netscience/email-class). "
            "Betweenness requires O(V) BFS passes; eigenvector requires convergent iteration."
        ),
        "pattern": "Multi-source BFS, power iteration, pinned convergence, register pressure management",
        "datasets": ["netscience.csv", "email-Eu-core.csv"],
    },
    "link_prediction": {
        "topology": "bipartite",
        "topology_desc": (
            "graph with bipartite tendencies (polbooks-class) or dense communities. "
            "Similarity computation requires efficient sorted-set intersection."
        ),
        "pattern": "Warp-cooperative set intersection, __shfl_sync for broadcast, vectorized memory loads",
        "datasets": ["polbooks.csv", "dolphins.csv"],
    },
    "cores": {
        "topology": "sparse-islands",
        "topology_desc": (
            "graph with heterogeneous density. K-core peeling iteratively removes low-degree vertices. "
            "Atomic degree tracking dominates; separate compilation needed for complex control flow."
        ),
        "pattern": "Peeling algorithm, atomic degree decrement, relocatable device code (--rdc=true)",
        "datasets": ["netscience.csv", "email-Eu-core.csv"],
    },
    "tree": {
        "topology": "sparse-islands",
        "topology_desc": (
            "weighted graph for minimum spanning tree (Borůvka's algorithm). "
            "Edge contraction requires efficient parallel prefix operations."
        ),
        "pattern": "Borůvka's algorithm, edge contraction via prefix scan, thrust library integration",
        "datasets": ["karate.csv", "netscience.csv"],
    },
}

# ── Algorithm name extraction ─────────────────────────────────────────────

ALGO_DISPLAY_NAMES: dict[str, str] = {
    "bfs": "Breadth-First Search (BFS)",
    "bfs_direction_optimizing": "Direction-Optimizing BFS",
    "bfs_mask": "BFS with Edge Masking",
    "bfs_seg": "Segmented BFS",
    "bfs_seg_mask": "Segmented BFS with Edge Masking",
    "bfs_direction_optimizing_mask": "Direction-Optimizing BFS with Masking",
    "bfs_direction_optimizing_seg": "Segmented Direction-Optimizing BFS",
    "bfs_direction_optimizing_seg_mask": "Segmented Direction-Optimizing BFS with Masking",
    "sssp_f32": "Single-Source Shortest Path (float32)",
    "weakly_connected_components": "Weakly Connected Components (WCC)",
    "strongly_connected_components": "Strongly Connected Components (SCC)",
    "pagerank": "PageRank",
    "hits": "HITS (Hubs and Authorities)",
    "louvain_f32": "Louvain Community Detection (float32)",
    "louvain_f64": "Louvain Community Detection (float64)",
    "leiden_f32": "Leiden Community Detection (float32)",
    "leiden_f64": "Leiden Community Detection (float64)",
    "triangle_count": "Triangle Counting",
    "k_truss": "K-Truss Decomposition",
    "ecg_f32": "Ensemble Clustering using Graph (float32)",
    "betweenness_centrality": "Betweenness Centrality",
    "edge_betweenness_centrality": "Edge Betweenness Centrality",
    "eigenvector_centrality": "Eigenvector Centrality",
    "katz_centrality": "Katz Centrality",
    "jaccard": "Jaccard Similarity",
    "cosine": "Cosine Similarity",
    "overlap": "Overlap Similarity",
    "sorensen": "Sørensen-Dice Similarity",
    "core_number": "Core Number Computation",
    "k_core": "K-Core Decomposition",
    "mst_f32": "Minimum Spanning Tree (float32)",
}


def _algo_base_name(filename: str) -> str:
    """Extract algorithm base name from filename, stripping variant suffixes."""
    name = filename.replace(".cu", "")
    # Remove common variant suffixes to get base algorithm
    for suffix in [
        "_seg_mask", "_seg", "_mask",
        "_f32_p32_seg_mask", "_f32_p32_seg", "_f32_p32_mask", "_f32_p32",
        "_f64_p64_seg_mask", "_f64_p64_seg", "_f64_p64_mask", "_f64_p64",
        "_p32_seg_mask", "_p32_seg", "_p32_mask", "_p32",
        "_p64_seg_mask", "_p64_seg", "_p64_mask", "_p64",
        "_f32_seg_mask", "_f32_seg", "_f32_mask",
        "_f64_seg_mask", "_f64_seg", "_f64_mask",
        "_all_pairs_f32_seg", "_all_pairs_f32", "_all_pairs_f64_seg", "_all_pairs_f64",
        "_all_pairs_seg", "_all_pairs",
    ]:
        if name.endswith(suffix) and len(name) > len(suffix):
            base = name[: -len(suffix)]
            if base:
                return base
    return name


def _variant_desc(filename: str) -> str:
    """Describe the variant from filename suffixes."""
    name = filename.replace(".cu", "")
    parts = []
    if "_f32" in name:
        parts.append("float32 precision")
    if "_f64" in name:
        parts.append("float64 precision")
    if "_p32" in name or "_p64" in name:
        parts.append("pinned memory optimization")
    if "_seg" in name:
        parts.append("segmented graph support")
    if "_mask" in name:
        parts.append("vertex masking")
    if "_all_pairs" in name:
        parts.append("all-pairs computation")
    if "_direction_optimizing" in name:
        parts.append("direction-optimizing (top-down/bottom-up switching)")
    return ", ".join(parts) if parts else "base implementation"


# ── Feature detection ─────────────────────────────────────────────────────

A100_FEATURE_PATTERNS = {
    "__launch_bounds__": re.compile(r"__launch_bounds__"),
    "__forceinline__": re.compile(r"__forceinline__"),
    "atomicCAS": re.compile(r"atomicCAS"),
    "atomicAdd": re.compile(r"atomicAdd"),
    "__ballot_sync": re.compile(r"__ballot_sync"),
    "__shfl_sync": re.compile(r"__shfl_sync"),
    "cudaHostAlloc": re.compile(r"cudaHostAlloc"),
    "cooperative_groups": re.compile(r"cooperative_groups"),
    "shared_memory": re.compile(r"__shared__"),
    "grid_stride_loop": re.compile(r"blockDim\.x \* gridDim\.x"),
    "thrust": re.compile(r"thrust::"),
    "cub": re.compile(r"cub::"),
    "warp_primitives": re.compile(r"__syncwarp|__shfl|__ballot"),
}


def _detect_features(code: str) -> list[str]:
    return [name for name, pat in A100_FEATURE_PATTERNS.items() if pat.search(code)]


def _detect_dispatch(code: str) -> str:
    if "cooperative_groups::this_grid" in code or "cg::this_grid" in code:
        return "cooperative"
    if "__syncwarp" in code and ("warp_id" in code or "lane" in code):
        return "warp"
    if "threadIdx.x" in code and "blockIdx.x" in code:
        return "thread"
    return "unknown"


# ── Read .cu.flags ────────────────────────────────────────────────────────

def _read_cu_flags(cu_path: str) -> list[str]:
    flags_path = cu_path + ".flags"
    if not os.path.exists(flags_path):
        return []
    with open(flags_path) as f:
        flags = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return flags


def _explain_flags(flags: list[str]) -> str:
    """Generate human-readable explanation for compilation flags."""
    explanations = []
    for flag in flags:
        if flag == "--use_fast_math":
            continue  # Universal, don't explain
        if flag.startswith("--maxrregcount="):
            n = flag.split("=")[1]
            explanations.append(
                f"Compile with `{flag}` to cap register usage at {n} per thread, "
                f"increasing SM occupancy for better latency hiding."
            )
        elif flag == "--rdc=true":
            explanations.append(
                f"Compile with `{flag}` (relocatable device code) for separate compilation — "
                f"needed when kernel uses cross-module device function calls."
            )
        elif flag == "--extra-device-vectorization":
            explanations.append(
                f"Compile with `{flag}` to auto-vectorize memory access patterns — "
                f"improves bandwidth utilization for memory-bound kernels."
            )
        elif "-Xptxas" in flag:
            explanations.append(f"Pass `{flag}` to PTX assembler for L1 cache policy control.")
        elif flag == "--expt-relaxed-constexpr":
            explanations.append(f"Use `{flag}` to allow constexpr in device code paths.")
    return " ".join(explanations)


# ── Prompt generation ─────────────────────────────────────────────────────

def _build_prompt(
    algo_base: str,
    variant: str,
    category: str,
    cu_flags: list[str],
    features: list[str],
) -> str:
    meta = CATEGORY_META.get(category, CATEGORY_META["traversal"])
    display_name = ALGO_DISPLAY_NAMES.get(algo_base, algo_base.replace("_", " ").title())

    prompt = (
        f"Write a high-performance CUDA kernel implementing {display_name} "
        f"for NVIDIA {TARGET_GPU} ({TARGET_ARCH}).\n\n"
    )

    prompt += f"**Target topology:** {meta['topology_desc']}\n\n"
    prompt += f"**Required A100 optimization pattern:** {meta['pattern']}\n\n"

    if variant != "base implementation":
        prompt += f"**Variant requirements:** {variant}\n\n"

    # Compilation flags
    non_default_flags = [f for f in cu_flags if f != "--use_fast_math"]
    if non_default_flags:
        flag_explanation = _explain_flags(cu_flags)
        prompt += f"**Compilation flags:** `{' '.join(cu_flags)}`\n"
        if flag_explanation:
            prompt += f"{flag_explanation}\n\n"
    elif cu_flags:
        prompt += f"**Compilation flags:** `{' '.join(cu_flags)}`\n\n"

    prompt += (
        "**Requirements:**\n"
        "- Input format: CSR arrays (offsets, indices, optionally weights, num_vertices)\n"
        "- Complete, compilable CUDA code targeting sm_80\n"
        "- Use grid-stride loops for scalability\n"
        "- Include `// CU_FLAGS:` comment with required compilation flags\n"
        "- Optimize for A100 hardware: 108 SMs, 40MB L2, 164KB SMEM/SM, 2TB/s HBM BW\n"
    )

    return prompt


# ── Main extraction ───────────────────────────────────────────────────────

def extract_all(a100_dir: str) -> list[dict[str, Any]]:
    """Walk the A100 impl directory and extract all kernels."""
    entries = []
    if not os.path.isdir(a100_dir):
        print(f"ERROR: A100 impl directory not found: {a100_dir}", file=sys.stderr)
        return entries

    for category_dir in sorted(Path(a100_dir).iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for cu_file in sorted(category_dir.glob("*.cu")):
            filename = cu_file.name
            cu_path = str(cu_file)

            # Read source
            with open(cu_path) as f:
                code = f.read()

            # Skip tiny files (likely just includes)
            if len(code.strip()) < 100:
                continue

            # Extract metadata
            algo_base = _algo_base_name(filename)
            variant = _variant_desc(filename)
            cu_flags = _read_cu_flags(cu_path)
            features = _detect_features(code)
            dispatch = _detect_dispatch(code)
            line_count = code.count("\n") + 1

            # Build prompt
            prompt = _build_prompt(algo_base, variant, category, cu_flags, features)

            # Format completion with CU_FLAGS comment
            flags_comment = ""
            if cu_flags:
                flags_comment = f"\n// CU_FLAGS: {' '.join(cu_flags)}"
            completion = f"```cuda\n{code}\n```{flags_comment}"

            entry = {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "algorithm": algo_base,
                    "variant": variant,
                    "cu_flags": cu_flags,
                    "category": category,
                    "line_count": line_count,
                    "target_topology": CATEGORY_META.get(category, {}).get("topology", "unknown"),
                    "a100_features": features,
                    "dispatch_type": dispatch,
                    "source_file": f"doubleGraph/cpp/src/aai/impl/a100/{category}/{filename}",
                    "datasets": CATEGORY_META.get(category, {}).get("datasets", []),
                },
            }
            entries.append(entry)

    return entries


def write_raw_dataset(entries: list[dict], output_path: str) -> None:
    """Write raw kernel dataset (prompt + completion + metadata)."""
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries to {output_path}")


def write_sft_dataset(entries: list[dict], output_path: str) -> None:
    """Write SFT dataset in HF messages format."""
    system_msg = (
        "You are a CUDA kernel optimization expert targeting NVIDIA A100 (sm_80). "
        "Write high-performance, compilable CUDA kernels. Include // CU_FLAGS comments "
        "for required compilation flags. Optimize for: 108 SMs, 40MB L2 cache, "
        "164KB shared memory per SM, 2 TB/s HBM bandwidth."
    )
    with open(output_path, "w") as f:
        for entry in entries:
            msg = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": entry["completion"]},
                ]
            }
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} SFT entries to {output_path}")


def write_grpo_prompts(entries: list[dict], output_path: str) -> None:
    """Write GRPO prompt-only dataset."""
    with open(output_path, "w") as f:
        for entry in entries:
            prompt_entry = {
                "prompt": entry["prompt"],
                "task_code": None,
                "data_source": "doublegraph_a100",
                "algorithm": entry["metadata"]["algorithm"],
                "category": entry["metadata"]["category"],
            }
            f.write(json.dumps(prompt_entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} GRPO prompts to {output_path}")


def print_stats(entries: list[dict]) -> None:
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"doubleGraph A100 Dataset Extraction Summary")
    print(f"{'='*60}")
    print(f"Total entries: {len(entries)}")

    by_category: dict[str, int] = {}
    by_topology: dict[str, int] = {}
    total_lines = 0
    flag_counts: dict[str, int] = {}

    for e in entries:
        m = e["metadata"]
        by_category[m["category"]] = by_category.get(m["category"], 0) + 1
        by_topology[m["target_topology"]] = by_topology.get(m["target_topology"], 0) + 1
        total_lines += m["line_count"]
        for flag in m["cu_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    print(f"Total source lines: {total_lines:,}")
    print(f"\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:25s} {count:4d}")
    print(f"\nBy topology:")
    for topo, count in sorted(by_topology.items()):
        print(f"  {topo:25s} {count:4d}")
    print(f"\nCompilation flags:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag:40s} {count:4d}")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract doubleGraph A100 kernels")
    parser.add_argument(
        "--doublegraph-root",
        default=DOUBLEGRAPH_ROOT,
        help="Path to doubleGraph repository root",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory for generated datasets",
    )
    args = parser.parse_args()

    a100_dir = os.path.join(args.doublegraph_root, "cpp", "src", "aai", "impl", "a100")
    print(f"Scanning: {a100_dir}")

    entries = extract_all(a100_dir)
    if not entries:
        print("ERROR: No kernels extracted!", file=sys.stderr)
        sys.exit(1)

    print_stats(entries)

    # Write all three dataset formats
    sft_path = os.path.join(args.output_dir, "doublegraph_sft.jsonl")

    write_sft_dataset(entries, sft_path)
    print(f"\nWrote {sft_path} ({len(entries)} entries)")
    print("Note: For RL training, use build_combined_dataset.py to generate combined_kernelforge.jsonl")


if __name__ == "__main__":
    main()
