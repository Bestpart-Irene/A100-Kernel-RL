"""Reward distribution monitoring for anti-reward-hacking validation.

Flags suspicious patterns that suggest the RL policy is gaming the reward
function rather than genuinely improving kernel quality.

Reward tiers (discrete milestones per CUDA Agent ablation):
  -1: compile or correctness failure
   1: correct but not faster
   2: faster than eager PyTorch (>5%)
   3: faster than torch.compile (>5%)

Reference: arXiv 2507.05619 (Detecting and Mitigating Reward Hacking)
"""
from __future__ import annotations

import math
from collections import Counter


def check_reward_distribution(rewards: list[float]) -> dict:
    """Analyze reward distribution for signs of reward hacking.

    Args:
        rewards: list of discrete reward values {-1, 1, 2, 3}

    Returns:
        Dict with distribution stats, entropy, and warning flags.
    """
    if not rewards:
        return {"distribution": {}, "flags": ["WARNING: empty reward list"], "entropy": 0.0}

    counts = Counter(rewards)
    total = len(rewards)
    flags = []

    # Flag: >90% at reward=3 — likely hacking (too-easy problems or gaming)
    if counts.get(3.0, 0) / total > 0.90:
        flags.append("SUSPICIOUS: >90% of rewards are at tier 3 (beats torch.compile)")

    # Flag: rewards collapse into failure/max modes only (no tier 1 or 2)
    if set(counts.keys()) <= {-1.0, 3.0} and len(counts) == 2:
        flags.append("WARNING: bimodal distribution (-1 vs 3 only, no intermediate tiers)")

    # Flag: all rewards identical — model collapsed
    if len(counts) == 1:
        flags.append("WARNING: uniform rewards — model has collapsed")

    # Flag: no positive rewards at all
    if all(r <= 0 for r in rewards):
        flags.append("WARNING: no positive rewards — model cannot generate correct kernels")

    # Flag: stuck at tier 1 (correct but never faster)
    if counts.get(1.0, 0) / total > 0.80 and total > 20:
        flags.append("INFO: >80% at tier 1 (correct but not faster) — consider easier problems or SFT")

    return {
        "distribution": dict(sorted(counts.items())),
        "total": total,
        "mean": sum(rewards) / total,
        "flags": flags,
        "entropy": _entropy(list(counts.values())),
        "tier_rates": {
            "fail_rate": sum(1 for r in rewards if r < 0) / total,
            "correct_rate": sum(1 for r in rewards if r >= 1.0) / total,
            "speedup_eager_rate": sum(1 for r in rewards if r >= 2.0) / total,
            "speedup_compile_rate": sum(1 for r in rewards if r >= 3.0) / total,
        },
    }


def _entropy(counts: list[int]) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)
