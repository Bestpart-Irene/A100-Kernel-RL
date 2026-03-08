"""EvoX: Self-evolving mutation strategies for AdaEvolve islands.

Implements the EvoX algorithm from SkyDiscover:
  - Hot-swappable strategy database: strategies can be added/removed at runtime
  - LogWindowScorer: tracks improvement velocity per strategy using sliding window
  - Stagnation detection: when a strategy stops improving, evolve it
  - Strategy evolution: combine elements from top-performing strategies

Designed to integrate with AdaEvolve — each island's strategy can be dynamically
adapted based on performance feedback.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Import strategy constants lazily to avoid circular import with adaevolve.py
# adaevolve imports EvoXStrategyManager; we need its STRATEGY_PROMPTS for hybrid generation.
_STRATEGY_PROMPTS: dict[str, str] | None = None


def _get_strategy_prompts() -> dict[str, str]:
    """Lazy import of STRATEGY_PROMPTS from adaevolve to break circular dependency."""
    global _STRATEGY_PROMPTS
    if _STRATEGY_PROMPTS is None:
        from skydiscover_integration.adaevolve import STRATEGY_PROMPTS
        _STRATEGY_PROMPTS = STRATEGY_PROMPTS
    return _STRATEGY_PROMPTS


@dataclass
class StrategyState:
    """Tracked state for a single mutation strategy."""
    name: str
    scores: deque = field(default_factory=lambda: deque(maxlen=20))
    total_evals: int = 0
    total_improvement: float = 0.0
    active: bool = True

    def record(self, score: float, prev_best: float) -> None:
        """Record an evaluation result for this strategy."""
        self.scores.append(score)
        self.total_evals += 1
        if score > prev_best:
            self.total_improvement += score - prev_best


class LogWindowScorer:
    """Track improvement velocity using log-scale sliding window.

    Measures how quickly a strategy is improving by looking at the slope
    of log(score) over a sliding window. Strategies with positive slope
    are improving; negative slope means degrading.
    """

    def __init__(self, window: int = 10):
        self.window = window

    def score(self, state: StrategyState) -> float:
        """Return improvement velocity (higher = faster improving).

        Uses log-scale to handle the wide range of kernel speedups
        (0.1x to 10x). Returns 0 for strategies with insufficient data.
        """
        scores = list(state.scores)
        if len(scores) < 3:
            return 0.0

        # Use last `window` scores
        recent = scores[-self.window:]

        # Compute log-scale improvement: mean of differences
        log_scores = []
        for s in recent:
            if s > 0:
                log_scores.append(math.log(s + 1))
            else:
                log_scores.append(-1.0)  # Penalty for failures

        if len(log_scores) < 2:
            return 0.0

        # Linear regression slope over the window
        n = len(log_scores)
        x_mean = (n - 1) / 2
        y_mean = sum(log_scores) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(log_scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Weight by recency — recent improvements matter more
        recency_weight = sum(log_scores[-3:]) / 3 if len(log_scores) >= 3 else 0

        return slope + 0.1 * recency_weight


# --- Strategy hybridization prompts ---

HYBRID_TEMPLATES: dict[str, str] = {
    "register_pressure+memory_coalescing": (
        "Optimize this CUDA kernel for REGISTER-EFFICIENT COALESCED ACCESS. Goals:\n"
        "- Use vectorized float4 loads (coalescing) while keeping register count low\n"
        "- Prefer shared memory over register-heavy local arrays\n"
        "- Use __launch_bounds__ to cap registers AND ensure coalesced access patterns\n"
        "- Balance: fewer registers per thread = more threads = better memory latency hiding"
    ),
    "memory_coalescing+warp_divergence": (
        "Optimize this CUDA kernel for DIVERGENCE-FREE COALESCED ACCESS. Goals:\n"
        "- Sort work items by size/type to group similar threads in warps\n"
        "- Use shared memory gather-scatter to regularize random access patterns\n"
        "- Replace if/else branches with predicated coalesced loads\n"
        "- Use warp shuffle to redistribute data without shared memory bank conflicts"
    ),
    "warp_divergence+occupancy_tuning": (
        "Optimize this CUDA kernel for HIGH-OCCUPANCY LOW-DIVERGENCE execution. Goals:\n"
        "- Tune block size to maximize active warps while minimizing divergence\n"
        "- Use warp-cooperative processing — each warp handles one work unit\n"
        "- Dynamic shared memory for flexible per-warp workspace\n"
        "- Target: >75% occupancy with <10% divergent instructions"
    ),
    "register_pressure+occupancy_tuning": (
        "Optimize this CUDA kernel for MAXIMUM OCCUPANCY. Goals:\n"
        "- Minimize registers per thread via __launch_bounds__ and --maxrregcount\n"
        "- Use smallest viable block size (128 or 256) to maximize blocks per SM\n"
        "- Trade ILP for occupancy — simpler code with more concurrent threads\n"
        "- Profile: target >80% theoretical occupancy on A100 (108 SMs)"
    ),
}


class EvoXStrategyManager:
    """Self-evolving mutation strategies for AdaEvolve islands.

    Tracks performance of each strategy, detects stagnation, and evolves
    new hybrid strategies by combining elements from top performers.

    Usage:
        manager = EvoXStrategyManager()
        strategy = manager.select_strategy(island_history)
        # ... use strategy for mutation ...
        manager.record_result(strategy, score, prev_best)
        if manager.check_stagnation(strategy):
            new_strategy = manager.evolve_strategy(strategy)
    """

    def __init__(
        self,
        initial_strategies: list[str] | None = None,
        stagnation_threshold: int = 10,
        window: int = 10,
    ):
        # Default strategies if none provided
        default = ["register_pressure", "memory_coalescing", "warp_divergence", "occupancy_tuning"]
        strategies = initial_strategies or default
        self.states: dict[str, StrategyState] = {
            s: StrategyState(name=s) for s in strategies
        }
        self.scorer = LogWindowScorer(window=window)
        self.stagnation_threshold = stagnation_threshold
        self.evolution_count = 0

    @property
    def active_strategies(self) -> list[str]:
        """List of currently active strategy names."""
        return [s for s, state in self.states.items() if state.active]

    def select_strategy(self, island_scores: list[float] | None = None) -> str:
        """Select the best strategy based on recent improvement velocity.

        If island_scores provided, uses them for context. Otherwise uses
        the LogWindowScorer to rank strategies by improvement velocity.
        """
        active = self.active_strategies
        if not active:
            return random.choice(list(self.states.keys()) or ["register_pressure"])

        # Score each active strategy
        scored = []
        for name in active:
            state = self.states[name]
            velocity = self.scorer.score(state)
            scored.append((name, velocity))

        # Softmax selection — bias toward high-velocity but allow exploration
        if not scored:
            return random.choice(active)

        max_score = max(s for _, s in scored)
        weights = [math.exp(s - max_score) for _, s in scored]
        total = sum(weights)
        if total == 0:
            return random.choice(active)

        r = random.random() * total
        cumulative = 0
        for (name, _), w in zip(scored, weights):
            cumulative += w
            if r <= cumulative:
                return name

        return scored[-1][0]

    def record_result(
        self, strategy: str, score: float, prev_best: float
    ) -> None:
        """Record an evaluation result for a strategy."""
        if strategy not in self.states:
            self.states[strategy] = StrategyState(name=strategy)
        self.states[strategy].record(score, prev_best)

    def check_stagnation(self, strategy: str) -> bool:
        """Check if a strategy has stagnated (no improvement in N evals)."""
        state = self.states.get(strategy)
        if not state:
            return False

        scores = list(state.scores)
        if len(scores) < self.stagnation_threshold:
            return False

        recent = scores[-self.stagnation_threshold:]
        # Stagnation: no improvement in the window
        if len(recent) < 2:
            return False

        best_early = max(recent[: len(recent) // 2])
        best_late = max(recent[len(recent) // 2:])

        return best_late <= best_early

    def evolve_strategy(self, stagnant_strategy: str) -> str:
        """Create a new hybrid strategy from top performers.

        When a strategy stagnates, combine it with the top-performing strategy
        to create a new hybrid. The hybrid gets a combined prompt that targets
        both optimization dimensions.
        """
        # Find top performer (excluding the stagnant one)
        best_name = None
        best_velocity = -float("inf")
        for name, state in self.states.items():
            if name == stagnant_strategy or not state.active:
                continue
            velocity = self.scorer.score(state)
            if velocity > best_velocity:
                best_velocity = velocity
                best_name = name

        if not best_name:
            # No other strategies — just reset the stagnant one
            self.states[stagnant_strategy].scores.clear()
            return stagnant_strategy

        # Create hybrid name
        self.evolution_count += 1
        pair_key = f"{stagnant_strategy}+{best_name}"
        reverse_key = f"{best_name}+{stagnant_strategy}"

        # Check if we have a template for this hybrid
        hybrid_prompt = HYBRID_TEMPLATES.get(
            pair_key, HYBRID_TEMPLATES.get(reverse_key)
        )

        if hybrid_prompt:
            hybrid_name = pair_key
        else:
            # Generate a generic hybrid prompt
            hybrid_name = f"hybrid_{self.evolution_count}"
            prompts = _get_strategy_prompts()
            prompt_a = prompts.get(stagnant_strategy, "")
            prompt_b = prompts.get(best_name, "")
            hybrid_prompt = (
                f"Combine two optimization strategies:\n\n"
                f"Strategy A ({stagnant_strategy}):\n{prompt_a}\n\n"
                f"Strategy B ({best_name}):\n{prompt_b}\n\n"
                f"Apply BOTH sets of optimizations to the kernel."
            )

        # Register hybrid strategy in the shared prompt registry
        prompts = _get_strategy_prompts()
        prompts[hybrid_name] = hybrid_prompt
        self.states[hybrid_name] = StrategyState(name=hybrid_name)

        # Deactivate stagnant strategy
        self.states[stagnant_strategy].active = False

        return hybrid_name

    def status(self) -> dict[str, Any]:
        """Return strategy manager status for logging."""
        return {
            "active_strategies": self.active_strategies,
            "evolution_count": self.evolution_count,
            "velocities": {
                name: self.scorer.score(state)
                for name, state in self.states.items()
                if state.active
            },
            "eval_counts": {
                name: state.total_evals
                for name, state in self.states.items()
            },
        }
