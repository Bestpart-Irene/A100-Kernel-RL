"""AdaEvolve: Multi-island evolutionary search with UCB scheduling for CUDA kernels.

Implements the AdaEvolve algorithm from SkyDiscover:
  - Multiple islands, each with a different mutation strategy
  - UCB1 (Upper Confidence Bound) scheduling allocates eval budget to promising islands
  - Per-island improvement signal (G) tracks which strategies are working
  - Paradigm breakthroughs: when an island finds a 2x+ improvement, broadcast to all

Uses the existing KernelForgeEvaluator bridge for evaluation:
  Stage 1: local nvcc compile check (fast, free)
  Stage 2: configured remote A100 benchmark (slow, backend-dependent cost)
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from skydiscover_integration.evaluator import EvaluationResult, KernelForgeEvaluator
from skydiscover_integration.evox_strategies import EvoXStrategyManager


# --- Mutation strategy definitions ---

MUTATION_STRATEGIES = [
    "register_pressure",
    "memory_coalescing",
    "warp_divergence",
    "occupancy_tuning",
]

# Strategy-specific mutation prompts for LLM-based kernel rewriting
STRATEGY_PROMPTS: dict[str, str] = {
    "register_pressure": (
        "Optimize this CUDA kernel for REGISTER PRESSURE. Goals:\n"
        "- Reduce register usage per thread to increase occupancy\n"
        "- Use __launch_bounds__(threads, minBlocks) to cap registers\n"
        "- Replace local arrays with shared memory where possible\n"
        "- Use --maxrregcount compiler flag if needed\n"
        "- Trade some ILP for lower register count"
    ),
    "memory_coalescing": (
        "Optimize this CUDA kernel for MEMORY COALESCING. Goals:\n"
        "- Ensure consecutive threads access consecutive memory addresses\n"
        "- Use float4/int4 vectorized loads for aligned data\n"
        "- Replace scatter patterns with shared memory gather-scatter\n"
        "- Use __ldg() for read-only data (texture cache path)\n"
        "- Align data structures to 128-byte boundaries"
    ),
    "warp_divergence": (
        "Optimize this CUDA kernel for WARP DIVERGENCE reduction. Goals:\n"
        "- Replace if/else branches with predicated execution where possible\n"
        "- Group threads with similar control flow into the same warps\n"
        "- Use warp-level primitives (__shfl, __ballot_sync) instead of branches\n"
        "- Sort work items by type before dispatch to reduce divergence\n"
        "- Use warp-cooperative processing for variable-length work"
    ),
    "occupancy_tuning": (
        "Optimize this CUDA kernel for OCCUPANCY. Goals:\n"
        "- Tune block size (128/256/512) to maximize active warps per SM\n"
        "- Balance shared memory usage vs thread count\n"
        "- Use dynamic shared memory allocation for flexibility\n"
        "- Ensure block count >= 2× number of SMs (108 for A100)\n"
        "- Profile: target >75% theoretical occupancy"
    ),
}


@dataclass
class Candidate:
    """A single kernel candidate in the population."""
    code: str
    score: float = -1.0
    generation: int = 0
    parent_id: str = ""
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"gen{self.generation}_{hash(self.code) % 100000:05d}"


@dataclass
class Island:
    """An island in the multi-island population with a fixed mutation strategy."""
    strategy: str
    population: list[Candidate] = field(default_factory=list)
    best_score: float = -1.0
    total_improvement: float = 0.0
    eval_count: int = 0
    stagnation_counter: int = 0
    max_population: int = 10

    def add(self, candidate: Candidate) -> None:
        """Add candidate to population, evict worst if at capacity."""
        self.population.append(candidate)
        if candidate.score > self.best_score:
            improvement = candidate.score - self.best_score
            self.best_score = candidate.score
            self.total_improvement += improvement
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.eval_count += 1

        # Evict worst if over capacity
        if len(self.population) > self.max_population:
            self.population.sort(key=lambda c: c.score, reverse=True)
            self.population = self.population[: self.max_population]

    def sample(self) -> Candidate:
        """Sample a parent from the population (tournament selection, k=2)."""
        if len(self.population) < 2:
            return self.population[0]
        a, b = random.sample(self.population, 2)
        return a if a.score >= b.score else b

    @property
    def improvement_rate(self) -> float:
        """Per-evaluation improvement rate (G signal)."""
        if self.eval_count == 0:
            return 0.0
        return self.total_improvement / self.eval_count


class AdaEvolve:
    """Multi-island evolutionary search using UCB scheduling.

    Each island has a different mutation strategy. UCB1 allocates the eval
    budget to the most promising islands. When an island finds a paradigm
    breakthrough (2x+ improvement), the candidate is broadcast to all islands.

    Usage:
        evaluator = KernelForgeEvaluator()
        seeds = [open(f).read() for f in seed_files]
        evo = AdaEvolve(evaluator, seeds, n_islands=4, budget=100)
        results = evo.run()

    The mutate() method is a stub that returns candidates with strategy-tagged
    comments. In production, wire this to the B200's LLM for intelligent
    mutation. For hackathon, simple regex-based mutations are used as fallback.
    """

    def __init__(
        self,
        evaluator: KernelForgeEvaluator,
        seeds: list[str],
        n_islands: int = 4,
        budget: int = 100,
        exploration_constant: float = 1.414,
        breakthrough_threshold: float = 2.0,
        output_dir: str = "outputs/adaevolve",
    ):
        self.evaluator = evaluator
        self.budget = budget
        self.exploration_constant = exploration_constant
        self.breakthrough_threshold = breakthrough_threshold
        self.output_dir = output_dir
        self.total_evals = 0
        self.breakthroughs: list[dict] = []
        self.log: list[dict] = []
        self.strategy_manager = EvoXStrategyManager(
            initial_strategies=MUTATION_STRATEGIES[:n_islands],
            stagnation_threshold=10,
        )

        # Initialize islands with strategies
        strategies = MUTATION_STRATEGIES[:n_islands]
        while len(strategies) < n_islands:
            strategies.append(random.choice(MUTATION_STRATEGIES))

        self.islands: list[Island] = []
        for strategy in strategies:
            seed_candidates = [
                Candidate(code=s, generation=0, strategy="seed")
                for s in seeds
            ]
            island = Island(strategy=strategy, population=seed_candidates)
            self.islands.append(island)

    def run(self) -> list[dict]:
        """Run evolutionary search for `budget` evaluations.

        Returns list of best kernels with scores and metadata.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"AdaEvolve: {len(self.islands)} islands, budget={self.budget}")

        for step in range(self.budget):
            # UCB1 island selection
            island_idx = self._ucb_select()
            island = self.islands[island_idx]

            # Select parent and mutate
            parent = island.sample()
            child_code = self._mutate(parent.code, island.strategy)
            child = Candidate(
                code=child_code,
                generation=parent.generation + 1,
                parent_id=parent.id,
                strategy=island.strategy,
            )

            # Evaluate: stage1 (local compile) → stage2 (remote A100 backend)
            stage1 = self.evaluator.evaluate_stage1(child_code)
            if stage1.combined_score > 0:
                stage2 = self.evaluator.evaluate_stage2(child_code)
                child.score = stage2.combined_score
                child.metadata = stage2.metrics
            else:
                child.score = stage1.combined_score
                child.metadata = {"compile_error": stage1.error}

            self.total_evals += 1

            # Add to island
            old_best = island.best_score
            island.add(child)

            # EvoX: record result and check for strategy stagnation
            self.strategy_manager.record_result(
                island.strategy, child.score, old_best
            )
            if self.strategy_manager.check_stagnation(island.strategy):
                new_strategy = self.strategy_manager.evolve_strategy(island.strategy)
                print(f"  EvoX: Island {island_idx} strategy evolved: "
                      f"{island.strategy} → {new_strategy}")
                island.strategy = new_strategy
                island.stagnation_counter = 0

            # Check for paradigm breakthrough
            if (
                child.score > 0
                and old_best > 0
                and child.score > self.breakthrough_threshold * old_best
            ):
                self._broadcast_breakthrough(child, island_idx)

            # Log progress
            self._log_step(step, island_idx, child)

            if (step + 1) % 10 == 0 or step == 0:
                self._print_status(step + 1)

        return self._collect_results()

    def _ucb_select(self) -> int:
        """UCB1 island selection — balance exploitation and exploration."""
        if self.total_evals == 0:
            return 0

        best_ucb = -float("inf")
        best_idx = 0

        for i, island in enumerate(self.islands):
            if island.eval_count == 0:
                return i  # Always try unvisited islands first

            # UCB1: mean reward + exploration bonus
            mean_reward = island.improvement_rate
            exploration = self.exploration_constant * math.sqrt(
                math.log(self.total_evals + 1) / island.eval_count
            )
            ucb = mean_reward + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        return best_idx

    def _mutate(self, kernel_code: str, strategy: str) -> str:
        """Apply mutation strategy to kernel code.

        In production, this should call the B200's LLM with a strategy-specific
        prompt to intelligently rewrite the kernel. For hackathon MVP, we use
        simple strategy-tagged comment injection + parameter tweaks.

        Override this method to wire in LLM-based mutation.
        """
        strategy_prompt = STRATEGY_PROMPTS.get(strategy, "")
        mutations = []

        if strategy == "register_pressure":
            mutations = [
                # Add launch bounds if not present
                ("__global__ void", "__global__ __launch_bounds__(256, 2) void"),
                # Suggest maxrregcount
                ("// CU_FLAGS:", f"// CU_FLAGS: --maxrregcount=64"),
            ]
        elif strategy == "memory_coalescing":
            mutations = [
                # Add ldg for read-only accesses
                ("= data[", "= __ldg(&data["),
            ]
        elif strategy == "warp_divergence":
            mutations = [
                # Add warp-level hint comment
                ("__global__", "// Strategy: minimize warp divergence\n__global__"),
            ]
        elif strategy == "occupancy_tuning":
            mutations = [
                # Try different block sizes
                ("blockDim.x", "blockDim.x"),  # noop — real mutation needs LLM
            ]

        mutated = kernel_code
        # Apply first applicable mutation
        for old, new in mutations:
            if old in mutated and new not in mutated:
                mutated = mutated.replace(old, new, 1)
                break

        # Add strategy tag
        if f"// AdaEvolve strategy: {strategy}" not in mutated:
            mutated = f"// AdaEvolve strategy: {strategy}\n{mutated}"

        return mutated

    def _broadcast_breakthrough(self, candidate: Candidate, source_island: int) -> None:
        """Paradigm breakthrough — share winning candidate to all islands."""
        self.breakthroughs.append({
            "step": self.total_evals,
            "source_island": source_island,
            "strategy": candidate.strategy,
            "score": candidate.score,
            "id": candidate.id,
        })
        print(
            f"  BREAKTHROUGH! Island {source_island} ({candidate.strategy}): "
            f"score={candidate.score:.3f} — broadcasting to all islands"
        )

        for i, island in enumerate(self.islands):
            if i != source_island:
                broadcast_candidate = Candidate(
                    code=candidate.code,
                    score=candidate.score,
                    generation=candidate.generation,
                    parent_id=candidate.id,
                    strategy=f"broadcast_from_{candidate.strategy}",
                    metadata={"broadcast_from_island": source_island},
                )
                island.add(broadcast_candidate)

    def _log_step(self, step: int, island_idx: int, candidate: Candidate) -> None:
        """Log step for analysis."""
        self.log.append({
            "step": step,
            "island": island_idx,
            "strategy": candidate.strategy,
            "score": candidate.score,
            "generation": candidate.generation,
            "timestamp": time.time(),
        })

    def _print_status(self, step: int) -> None:
        """Print progress summary."""
        island_summaries = []
        for i, island in enumerate(self.islands):
            island_summaries.append(
                f"  Island {i} ({island.strategy}): "
                f"best={island.best_score:.3f}, "
                f"evals={island.eval_count}, "
                f"G={island.improvement_rate:.4f}, "
                f"stag={island.stagnation_counter}"
            )
        print(f"\n--- Step {step}/{self.budget} ---")
        for s in island_summaries:
            print(s)
        if self.breakthroughs:
            print(f"  Breakthroughs: {len(self.breakthroughs)}")

    def _collect_results(self) -> list[dict]:
        """Collect best kernels from all islands."""
        results = []
        seen_codes = set()

        for i, island in enumerate(self.islands):
            for candidate in sorted(
                island.population, key=lambda c: c.score, reverse=True
            ):
                code_hash = hash(candidate.code)
                if code_hash in seen_codes:
                    continue
                seen_codes.add(code_hash)

                results.append({
                    "code": candidate.code,
                    "score": candidate.score,
                    "island": i,
                    "strategy": candidate.strategy,
                    "generation": candidate.generation,
                    "metrics": candidate.metadata,
                })

        results.sort(key=lambda r: r["score"], reverse=True)

        # Save results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "total_evals": self.total_evals,
                    "breakthroughs": self.breakthroughs,
                    "best_score": results[0]["score"] if results else 0,
                    "results": results[:20],  # Top 20
                },
                f,
                indent=2,
            )

        # Save log
        log_path = os.path.join(self.output_dir, "evolution_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log, f, indent=2)

        print(f"\nAdaEvolve complete. {self.total_evals} evaluations.")
        print(f"  Best score: {results[0]['score']:.3f}" if results else "  No valid results.")
        print(f"  Results saved to {results_path}")
        print(f"  Breakthroughs: {len(self.breakthroughs)}")

        return results
