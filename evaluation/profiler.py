from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

from evaluation.verifier import verify_kernel
from verification.pac_verify import generate_test_graphs, run_kernel_verification


@dataclass(slots=True)
class ProfileResult:
    runtime_ms: float
    runtime_stats: dict[str, float]
    verified: bool
    verifier_msg: str
    samples: int
    metadata: dict[str, Any]


def profile_kernel(
    so_path: str,
    warmup_iters: int = 20,
    benchmark_runs: int = 10,
    num_vertices: int = 10000,
) -> ProfileResult:
    verification = verify_kernel(so_path=so_path, num_vertices=num_vertices)
    if not verification.correct:
        return ProfileResult(
            runtime_ms=0.0,
            runtime_stats={},
            verified=False,
            verifier_msg=verification.verifier_msg,
            samples=0,
            metadata=verification.metadata,
        )

    graph_type, edges, n_verts = generate_test_graphs(num_vertices=num_vertices)[0]

    for _ in range(warmup_iters):
        run_kernel_verification(so_path, edges, n_verts)

    times_ms: list[float] = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        run_kernel_verification(so_path, edges, n_verts)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return ProfileResult(
        runtime_ms=statistics.median(times_ms),
        runtime_stats={
            "mean": statistics.fmean(times_ms),
            "median": statistics.median(times_ms),
            "min": min(times_ms),
            "max": max(times_ms),
            "std": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        },
        verified=True,
        verifier_msg=verification.verifier_msg,
        samples=len(times_ms),
        metadata={**verification.metadata, "graph_type": graph_type},
    )
