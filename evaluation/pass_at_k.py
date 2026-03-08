"""Pass@k metric — unbiased estimator for code generation evaluation.

Reference: Chen et al. 2021, "Evaluating Large Language Models Trained on Code"
https://arxiv.org/abs/2107.03374

For each problem, generate n samples, count c correct ones, compute pass@k.
Standard reporting: pass@1, pass@5, pass@10.
"""
from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Args:
        n: total number of generated samples per problem
        c: number of correct samples (passed all tests)
        k: k value for pass@k

    Returns:
        Probability that at least 1 of k random samples is correct.
    """
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c < 0 or c > n:
        raise ValueError(f"c ({c}) must be in [0, n={n}]")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_problems(results: list[dict], k_values: list[int] = None) -> dict:
    """Compute pass@k across multiple problems.

    Args:
        results: list of dicts with keys "n" (total samples) and "c" (correct samples)
        k_values: list of k values to compute (default: [1, 5, 10])

    Returns:
        Dict mapping k -> mean pass@k across all problems.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    output = {}
    for k in k_values:
        scores = []
        for r in results:
            n, c = r["n"], r["c"]
            if n >= k:
                scores.append(pass_at_k(n, c, k))
        output[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
        output[f"pass@{k}_count"] = len(scores)

    return output
