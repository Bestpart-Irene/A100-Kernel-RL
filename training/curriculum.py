"""
Curriculum learning for progressive CUDA kernel difficulty.

4 phases based on DoubleGraph's 4-way dispatch pattern:
  1. single_ops    — single operator kernels (vector_add, relu, softmax)
  2. fusion_2op    — 2-op fusions (83.77% of Ops-6K, the sweet spot)
  3. arch_specific — A100-specific optimizations (L2 pinning, cooperative kernels)
  4. advanced      — multi-op fusion, complex memory patterns

Promotion: >50% of last 10 rewards hit target → advance
Demotion:  <20% of last 10 rewards are positive (>0) → regress
"""
from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")

WINDOW_SIZE = 10
PROMOTE_THRESHOLD = 0.50   # >50% hit target reward → advance
DEMOTE_THRESHOLD = 0.20    # <20% positive → regress


@dataclass
class CurriculumPhase:
    """A single phase of the curriculum."""
    name: str
    target_reward: float
    problems: list[dict] = field(default_factory=list)
    description: str = ""


def _default_phases() -> list[CurriculumPhase]:
    """Build the 4-phase curriculum with built-in fallback problems."""
    return [
        CurriculumPhase(
            name="single_ops",
            target_reward=1.0,
            description="Single operator kernels — establish CUDA syntax and correctness",
            problems=[
                {
                    "prompt": f"Write a CUDA kernel for vector addition optimized for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Take float arrays A, B, output C, and length N.",
                    "ops": ["vector_add"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA ReLU activation kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Apply max(0, x) element-wise to a float array.",
                    "ops": ["relu"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA softmax kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute row-wise softmax for a 2D float matrix.",
                    "ops": ["softmax"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA matrix multiplication kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute C = A @ B for float matrices using shared memory tiling.",
                    "ops": ["matmul"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA kernel for element-wise GELU activation for {TARGET_GPU} ({TARGET_ARCH}).",
                    "ops": ["gelu"],
                    "difficulty": 1,
                },
            ],
        ),
        CurriculumPhase(
            name="fusion_2op",
            target_reward=2.0,
            description="2-op fusions — the Ops-6K sweet spot (83.77% of dataset)",
            problems=[
                {
                    "prompt": f"Write a fused CUDA kernel for LayerNorm + ReLU for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Combine normalization and activation in a single kernel launch.",
                    "ops": ["layer_norm", "relu"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for MatMul + BiasAdd for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute C = A @ B + bias in a single kernel.",
                    "ops": ["matmul", "bias_add"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for Softmax + Dropout for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Apply softmax then probabilistic dropout in one pass.",
                    "ops": ["softmax", "dropout"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for Conv2d + BatchNorm for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Fuse the convolution output directly into batch normalization.",
                    "ops": ["conv2d", "batch_norm"],
                    "difficulty": 2,
                },
            ],
        ),
        CurriculumPhase(
            name="arch_specific",
            target_reward=2.0,
            description="A100-specific optimizations — graph algorithms + L2 pinning + cooperative kernels",
            problems=[
                {
                    "prompt": f"Write a CUDA WCC (Weakly Connected Components) kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                              "using non-atomic Union-Find with L2 cache pinning for the parent array.",
                    "ops": ["wcc"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA reduction kernel for {TARGET_GPU} ({TARGET_ARCH}) using "
                              "cooperative groups grid-wide sync for single-kernel-launch reduction.",
                    "ops": ["reduce_sum"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA GEMM kernel for {TARGET_GPU} ({TARGET_ARCH}) using "
                              "shared memory double-buffering and vectorized float4 loads for maximum throughput.",
                    "ops": ["gemm"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA attention kernel for {TARGET_GPU} ({TARGET_ARCH}) that tiles "
                              "Q, K, V in shared memory to minimize HBM traffic. Target occupancy >75%.",
                    "ops": ["attention"],
                    "difficulty": 3,
                },
                # --- Topology-aware graph problems from doubleGraph ---
                {
                    "prompt": f"Write a BFS kernel for {TARGET_GPU} ({TARGET_ARCH}). The target is a "
                              "power-law web graph (HiBench-class, skewed degree distribution with hub vertices). "
                              "Standard per-vertex parallelism causes warp divergence on hubs. Implement "
                              "direction-optimizing traversal with bitmap frontier to switch between top-down "
                              "(queue-based for sparse frontiers) and bottom-up (bitmap scan for dense frontiers). "
                              "Use `--use_fast_math` and `__ldg()` for read-only graph data.",
                    "ops": ["bfs"],
                    "difficulty": 3,
                    "topology": "power-law",
                    "data_source": "doublegraph_a100",
                    "graph_properties": {
                        "type": "power-law",
                        "num_vertices": 10000,
                        "num_edges": 62000,
                        "avg_degree": 12.4,
                        "max_degree": 4891,
                        "density": 0.0012,
                        "diameter": 8,
                        "degree_distribution": "heavy-tail, exponent ~2.1",
                        "optimization_hints": [
                            "Hub vertices (degree>100) dominate traversal — warp-level processing for hubs",
                            "Long-tail low-degree vertices (degree<5, ~60% of graph) — thread-per-vertex",
                            "Frontier size varies 100x between BFS levels — direction-optimizing switch",
                            "CSR row_ptr access is sequential; col_idx is random → __ldg() for col_idx",
                        ],
                    },
                },
                {
                    "prompt": f"Write a PageRank kernel for {TARGET_GPU} ({TARGET_ARCH}). Target: email network "
                              "(dense, hierarchical, ~25K edges). Implement SpMV + sum-reduce with pinned memory "
                              "convergence check via `cudaHostAlloc` + `cudaHostGetDevicePointer` for zero-copy "
                              "convergence flag. Use float32 with `--use_fast_math`. The kernel should converge "
                              "within epsilon=1e-6 without a `cudaMemcpy` per iteration.",
                    "ops": ["pagerank"],
                    "difficulty": 3,
                    "topology": "dense-regular",
                    "data_source": "doublegraph_a100",
                    "graph_properties": {
                        "type": "dense-regular",
                        "num_vertices": 1005,
                        "num_edges": 25571,
                        "avg_degree": 25.4,
                        "max_degree": 71,
                        "density": 0.0253,
                        "diameter": 5,
                        "degree_distribution": "near-uniform, slight hierarchy (email org structure)",
                        "optimization_hints": [
                            "Uniform degree → balanced warp workload, no hub/leaf divergence",
                            "Dense rows (avg 25 neighbors) → vectorized float4 loads for SpMV",
                            "Small graph fits in L2 cache (40MB on A100) — pin adjacency in L2",
                            "Convergence check via zero-copy avoids cudaMemcpy per iteration",
                        ],
                    },
                },
                {
                    "prompt": f"Write a WCC kernel for {TARGET_GPU} ({TARGET_ARCH}) targeting sparse, disconnected "
                              "networks (netscience-class, ~1.5K vertices, many isolated components). Use "
                              "non-atomic path-halving Union-Find: `p = parent[parent[p]]` without atomics. "
                              "Convergence via zero-copy flag. Include sampled hooking (k=2 neighbors first) "
                              "then full scan. Compile with `--use_fast_math --extra-device-vectorization`.",
                    "ops": ["wcc"],
                    "difficulty": 3,
                    "topology": "sparse-islands",
                    "data_source": "doublegraph_a100",
                    "graph_properties": {
                        "type": "sparse-islands",
                        "num_vertices": 1589,
                        "num_edges": 2742,
                        "avg_degree": 3.45,
                        "max_degree": 34,
                        "density": 0.0022,
                        "diameter": 17,
                        "degree_distribution": "sparse with isolated clusters, many degree-1 vertices",
                        "optimization_hints": [
                            "Many isolated components → early termination when convergence flag is set",
                            "Low degree → thread-per-vertex dispatch (no warp-level needed)",
                            "Sparse random access → __ldg() for read-only parent array",
                            "Small working set → fits entirely in L2, pin parent[] array",
                        ],
                    },
                },
            ],
        ),
        CurriculumPhase(
            name="advanced",
            target_reward=3.0,
            description="Multi-op fusion, complex graph kernels — beat torch.compile and cuGraph",
            problems=[
                {
                    "prompt": f"Write a fused CUDA kernel for LayerNorm + GELU + Linear for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Fuse all three operations to minimize memory round-trips.",
                    "ops": ["layer_norm", "gelu", "linear"],
                    "difficulty": 4,
                },
                {
                    "prompt": f"Write a fused Flash-Attention-style kernel for {TARGET_GPU} ({TARGET_ARCH}) that "
                              "computes Q @ K^T, applies causal mask, softmax, and V multiply in one kernel.",
                    "ops": ["flash_attention"],
                    "difficulty": 4,
                },
                {
                    "prompt": f"Write a CUDA kernel for batched sparse matrix-vector multiplication (SpMV) "
                              f"for {TARGET_GPU} ({TARGET_ARCH}), using CSR format with warp-level load balancing.",
                    "ops": ["batched_spmv"],
                    "difficulty": 4,
                },
                # --- Advanced graph problems from doubleGraph ---
                {
                    "prompt": f"Write a triangle counting kernel for {TARGET_GPU} ({TARGET_ARCH}). Target: social "
                              "network with community clustering. Use set intersection via sorted merge with "
                              "warp-level binary search. Each warp handles one vertex's neighbor list intersection. "
                              "Compile with `--use_fast_math --maxrregcount=64` to balance register pressure. "
                              "Expected: warp-cooperative processing for high-degree vertices.",
                    "ops": ["triangle_count"],
                    "difficulty": 4,
                    "topology": "dense-community",
                    "data_source": "doublegraph_a100",
                    "graph_properties": {
                        "type": "dense-community",
                        "num_vertices": 4039,
                        "num_edges": 88234,
                        "avg_degree": 43.7,
                        "max_degree": 1045,
                        "density": 0.0108,
                        "diameter": 8,
                        "clustering_coefficient": 0.6055,
                        "degree_distribution": "community-structured with high-degree hubs per community",
                        "optimization_hints": [
                            "High clustering → many triangles per vertex, O(d^2) work for hubs",
                            "Sorted neighbor lists enable merge-based set intersection",
                            "Warp-level binary search: each warp lane checks one candidate",
                            "Register pressure is bottleneck — cap at 64 regs for occupancy",
                            "Hub vertices (degree>100) need warp-cooperative processing",
                        ],
                    },
                },
                {
                    "prompt": f"Write a Louvain community detection kernel for {TARGET_GPU} ({TARGET_ARCH}). Target: "
                              "social network with distinct community clustering (Karate/Dolphins-class). The local "
                              "move phase must use warp-level shared-memory hash tables (128 entries per warp) to "
                              "accumulate inter-community edge weights. Dispatch: warp-cooperative for degree>=8, "
                              "thread-per-vertex for sparse. Use `--use_fast_math --maxrregcount=48` to increase "
                              "occupancy for this latency-bound algorithm.",
                    "ops": ["louvain"],
                    "difficulty": 4,
                    "topology": "dense-community",
                    "data_source": "doublegraph_a100",
                    "graph_properties": {
                        "type": "dense-community",
                        "num_vertices": 34,
                        "num_edges": 156,
                        "avg_degree": 9.2,
                        "max_degree": 17,
                        "density": 0.139,
                        "diameter": 5,
                        "clustering_coefficient": 0.5706,
                        "num_communities": 4,
                        "degree_distribution": "bimodal — core members (degree>12) and periphery (degree<6)",
                        "optimization_hints": [
                            "Small graph but algorithm pattern matters for scaling",
                            "Bimodal degree → dual dispatch: warp-cooperative for degree>=8, thread for rest",
                            "Shared-memory hash tables (128 entries/warp) for community edge weight accumulation",
                            "Modularity delta computation is latency-bound → maximize occupancy via low reg count",
                            "4 ground-truth communities — convergence in 2-3 Louvain passes",
                        ],
                    },
                },
            ],
        ),
    ]


def format_topology_context(problem: dict) -> str:
    """Format graph topology properties as text context for the model.

    When a curriculum problem includes graph_properties, this produces a
    structured context block that teaches the model WHY certain CUDA patterns
    are optimal for the given topology. This is the key RL signal that makes
    topology-aware optimization possible — the model sees graph structure, not
    just text descriptions.

    Returns empty string for non-topology problems (phases 0-1, generic ops).
    """
    props = problem.get("graph_properties")
    if not props:
        return ""

    lines = [
        "\n## Graph Topology Context",
        f"Type: {props.get('type', 'unknown')}",
        f"Vertices: {props.get('num_vertices', '?'):,} | Edges: {props.get('num_edges', '?'):,}",
        f"Avg degree: {props.get('avg_degree', '?')} | Max degree: {props.get('max_degree', '?')}",
        f"Density: {props.get('density', '?')} | Diameter: {props.get('diameter', '?')}",
        f"Distribution: {props.get('degree_distribution', 'unknown')}",
    ]

    if props.get("clustering_coefficient"):
        lines.append(f"Clustering coefficient: {props['clustering_coefficient']}")
    if props.get("num_communities"):
        lines.append(f"Communities: {props['num_communities']}")

    hints = props.get("optimization_hints", [])
    if hints:
        lines.append("\nOptimization guidance (from A100 expert kernels):")
        for hint in hints:
            lines.append(f"  - {hint}")

    return "\n".join(lines)


def format_problem_prompt(problem: dict) -> str:
    """Build a complete prompt from a curriculum problem dict.

    Combines the base prompt with topology context (if available).
    Use this instead of raw problem["prompt"] to get topology-aware prompts.
    """
    base = problem.get("prompt", "")
    topo_ctx = format_topology_context(problem)
    if topo_ctx:
        return f"{base}\n{topo_ctx}"
    return base


class CurriculumManager:
    """Manages progressive difficulty for CUDA kernel training."""

    def __init__(self, phases: list[CurriculumPhase] | None = None):
        self.phases = phases or _default_phases()
        self.current_phase_idx = 0
        self.reward_history: deque[float] = deque(maxlen=WINDOW_SIZE)
        self.phase_history: list[dict] = []  # track promotions/demotions

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_phase_idx]

    @property
    def phase_name(self) -> str:
        return self.current_phase.name

    def get_problem(self) -> dict:
        """Sample a random problem from the current phase."""
        problems = self.current_phase.problems
        if not problems:
            return {"prompt": f"Write a CUDA kernel optimized for {TARGET_GPU} ({TARGET_ARCH}).", "ops": [], "difficulty": 1}
        return random.choice(problems)

    def record_reward(self, reward: float) -> str | None:
        """Record a reward and check for promotion/demotion.

        Returns:
            "promoted", "demoted", or None if no change.
        """
        self.reward_history.append(reward)

        if len(self.reward_history) < WINDOW_SIZE:
            return None

        rewards = list(self.reward_history)
        target = self.current_phase.target_reward

        # Check promotion: >50% of recent rewards hit target
        hit_rate = sum(1 for r in rewards if r >= target) / len(rewards)
        if hit_rate > PROMOTE_THRESHOLD and self.current_phase_idx < len(self.phases) - 1:
            self._advance()
            return "promoted"

        # Check demotion: <20% of recent rewards are positive
        positive_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        if positive_rate < DEMOTE_THRESHOLD and self.current_phase_idx > 0:
            self._regress()
            return "demoted"

        return None

    def _advance(self):
        old = self.current_phase.name
        self.current_phase_idx += 1
        self.reward_history.clear()
        self.phase_history.append({
            "action": "promoted",
            "from": old,
            "to": self.current_phase.name,
        })
        print(f"Curriculum: PROMOTED {old} → {self.current_phase.name}")

    def _regress(self):
        old = self.current_phase.name
        self.current_phase_idx -= 1
        self.reward_history.clear()
        self.phase_history.append({
            "action": "demoted",
            "from": old,
            "to": self.current_phase.name,
        })
        print(f"Curriculum: DEMOTED {old} → {self.current_phase.name}")

    def add_problems(self, phase_name: str, problems: list[dict]):
        """Add external problems (e.g. from Ops-6K) to a phase."""
        for phase in self.phases:
            if phase.name == phase_name:
                phase.problems.extend(problems)
                return
        raise ValueError(f"Unknown phase: {phase_name}")

    def status(self) -> dict:
        """Return current curriculum status."""
        rewards = list(self.reward_history)
        target = self.current_phase.target_reward
        return {
            "phase": self.current_phase.name,
            "phase_idx": self.current_phase_idx,
            "total_phases": len(self.phases),
            "target_reward": target,
            "window_size": len(rewards),
            "hit_rate": sum(1 for r in rewards if r >= target) / len(rewards) if rewards else 0,
            "positive_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0,
            "transitions": len(self.phase_history),
        }
