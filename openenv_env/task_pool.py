"""Task pool manager for the KernelForge RL environment.

Loads tasks from JSONL pool files, samples tasks for reset(), and
caches baselines per task to avoid redundant A100 evaluation calls.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_POOL_PATH = _PROJECT_ROOT / "tasks" / "pool_v0.jsonl"


class TaskPool:
    """Manages a pool of evaluable tasks for RL episodes.

    Usage:
        pool = TaskPool.load()           # Load from default JSONL
        task = pool.sample()             # Random task
        task = pool.sample(task_id=...) # Specific task
    """

    def __init__(self, tasks: list[dict[str, Any]]):
        self.tasks = tasks
        self._by_id: dict[str, dict[str, Any]] = {}
        for t in tasks:
            tid = t.get("task_id", "")
            if tid:
                self._by_id[tid] = t
        self._baseline_cache: dict[str, dict[str, float]] = {}

    @classmethod
    def load(cls, pool_path: str | Path | None = None) -> TaskPool:
        """Load task pool from a JSONL file.

        Falls back to the combined_kernelforge.jsonl dataset if no pool file exists.
        """
        path = Path(pool_path) if pool_path else _DEFAULT_POOL_PATH
        tasks: list[dict[str, Any]] = []

        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
        else:
            # Fallback: load from combined dataset, filter to supported
            combined = _PROJECT_ROOT / "datasets" / "combined_kernelforge.jsonl"
            if combined.exists():
                with open(combined, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line.strip())
                        if row.get("evaluation_backend") in {"ops6k", "wcc"}:
                            tasks.append(row)

        if not tasks:
            # Last resort: create a minimal default task set
            tasks = _builtin_tasks()

        return cls(tasks)

    def sample(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Sample a task from the pool.

        Args:
            task_id: If provided, return this specific task.
            seed: Random seed for reproducible sampling.
            backend: If provided, filter to tasks with this backend ("ops6k" or "wcc").
        """
        if task_id and task_id in self._by_id:
            return dict(self._by_id[task_id])

        pool = self.tasks
        if backend:
            pool = [t for t in pool if t.get("evaluation_backend") == backend]

        if not pool:
            pool = self.tasks

        rng = random.Random(seed) if seed is not None else random
        return dict(rng.choice(pool))

    def get_cached_baselines(self, task_id: str) -> dict[str, float] | None:
        """Return cached baselines for a task, or None."""
        return self._baseline_cache.get(task_id)

    def cache_baselines(self, task_id: str, baselines: dict[str, float]) -> None:
        """Cache baselines for a task."""
        self._baseline_cache[task_id] = baselines

    @property
    def size(self) -> int:
        return len(self.tasks)

    @property
    def ops6k_count(self) -> int:
        return sum(1 for t in self.tasks if t.get("evaluation_backend") == "ops6k")

    @property
    def wcc_count(self) -> int:
        return sum(1 for t in self.tasks if t.get("evaluation_backend") == "wcc")

    def summary(self) -> dict[str, Any]:
        """Return a summary of the task pool."""
        from collections import Counter

        backends = Counter(t.get("evaluation_backend", "unknown") for t in self.tasks)
        sources = Counter(t.get("data_source", "unknown") for t in self.tasks)
        return {
            "total": self.size,
            "backends": dict(backends),
            "sources": dict(sources),
            "cached_baselines": len(self._baseline_cache),
        }


def _builtin_tasks() -> list[dict[str, Any]]:
    """Minimal built-in tasks when no pool file exists."""
    return [
        {
            "task_id": "builtin_wcc_001",
            "prompt": "Optimize a Weakly Connected Components CUDA kernel.",
            "ops": ["weakly_connected_components"],
            "data_source": "builtin",
            "evaluation_backend": "wcc",
            "supports_evaluation": True,
            "difficulty": 1,
        },
        {
            "task_id": "builtin_elu_001",
            "prompt": (
                "Write a high-performance CUDA kernel for NVIDIA A100 (sm_80) "
                "that implements F.elu (Exponential Linear Unit) on a tensor."
            ),
            "ops": ["F.elu"],
            "data_source": "builtin",
            "evaluation_backend": "ops6k",
            "supports_evaluation": True,
            "task_code": _BUILTIN_ELU_CODE,
            "difficulty": 1,
        },
    ]


_BUILTIN_ELU_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.elu(x, alpha=1.0)

def get_inputs():
    return [torch.randn(16, 1024, 1024)]

def get_init_inputs():
    return []
""".strip()
