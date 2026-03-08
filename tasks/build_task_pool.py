#!/usr/bin/env python3
"""Build a validated task pool from CUDA-Agent-Ops-6K for RL training.

Loads the Ops-6K dataset from HuggingFace, filters for stateless tasks
that our extension harness can evaluate, and outputs a JSONL pool file.

Usage:
    python tasks/build_task_pool.py                    # Build pool
    python tasks/build_task_pool.py --validate         # Validate existing pool
    python tasks/build_task_pool.py --max-samples 2000 # Limit source samples
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.cuda_agent_integration import _parse_ops  # noqa: E402


def _task_has_empty_init_inputs(task_code: str) -> bool:
    """Return True when get_init_inputs is absent or returns []."""
    import ast

    if not task_code.strip():
        return False
    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "get_init_inputs":
            continue
        returns = [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Return)]
        if not returns:
            return True
        value = returns[-1]
        if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 0:
            return True
        if isinstance(value, ast.Constant) and value.value is None:
            return True
        return False
    return True


STATEFUL_TOKENS = (
    "nn.Conv",
    "nn.Linear",
    "nn.BatchNorm",
    "nn.LSTM",
    "nn.GRU",
    "nn.Embedding",
    "nn.Parameter",
    "nn.MultiheadAttention",
    "register_buffer(",
    "register_parameter(",
)

# Ops that are stochastic or hard to verify
DENY_OPS = {
    "torch.dropout", "F.dropout", "nn.Dropout",
    "torch.bernoulli", "torch.multinomial",
    "torch.randperm", "torch.rand", "torch.randn",
}


def is_stateless_evaluable(task_code: str) -> bool:
    """Return True for tasks that the ops6k extension harness can evaluate."""
    text = task_code.strip()
    if not text:
        return False
    if any(token in text for token in STATEFUL_TOKENS):
        return False
    if any(deny in text for deny in DENY_OPS):
        return False
    if "class Model" not in text:
        return False
    if "def forward" not in text:
        return False
    if "def get_inputs" not in text:
        return False
    return _task_has_empty_init_inputs(text)


def has_valid_structure(task_code: str) -> bool:
    """Check that the task code can be parsed and has required components."""
    import ast

    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False

    has_model = False
    has_get_inputs = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            has_model = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            has_get_inputs = True
    return has_model and has_get_inputs


def build_task_pool(max_samples: int = 6000, seed: int = 42) -> list[dict]:
    """Load Ops-6K and filter for evaluable tasks."""
    # Import HF datasets (using the shadow-safe loader)
    from training.cuda_agent_integration import _load_hf_datasets

    hf_datasets = _load_hf_datasets()
    print(f"Loading CUDA-Agent-Ops-6K (max {max_samples} samples)...")
    ds = hf_datasets.load_dataset(
        "BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train"
    )
    print(f"  Loaded {len(ds)} total samples")

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        print(f"  Subsampled to {max_samples}")

    pool = []
    skipped = {"no_code": 0, "stateful": 0, "bad_structure": 0}

    for idx, row in enumerate(ds):
        code = str(row.get("code", "")).strip()
        if not code:
            skipped["no_code"] += 1
            continue

        if not is_stateless_evaluable(code):
            skipped["stateful"] += 1
            continue

        if not has_valid_structure(code):
            skipped["bad_structure"] += 1
            continue

        ops = _parse_ops(row.get("ops"))
        ops_desc = ", ".join(ops) if ops else "unknown"

        pool.append({
            "task_id": f"ops6k_{idx:05d}",
            "task_code": code,
            "ops": ops,
            "ops_description": ops_desc,
            "data_source": str(row.get("data_source", "ops6k")),
            "evaluation_backend": "ops6k",
            "supports_evaluation": True,
        })

    print(f"\nFilter results:")
    print(f"  Evaluable: {len(pool)}")
    print(f"  Skipped (no code): {skipped['no_code']}")
    print(f"  Skipped (stateful): {skipped['stateful']}")
    print(f"  Skipped (bad structure): {skipped['bad_structure']}")

    return pool


def save_pool(pool: list[dict], output_path: Path) -> None:
    """Save pool as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in pool:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"Saved {len(pool)} tasks to {output_path}")


def validate_pool(pool_path: Path) -> None:
    """Validate that pool tasks have correct structure."""
    if not pool_path.exists():
        print(f"Pool file not found: {pool_path}")
        return

    with open(pool_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Validating {len(tasks)} tasks from {pool_path}...")
    valid = 0
    invalid = 0
    for task in tasks:
        code = task.get("task_code", "")
        if is_stateless_evaluable(code) and has_valid_structure(code):
            valid += 1
        else:
            invalid += 1
            print(f"  INVALID: {task.get('task_id', '?')} ops={task.get('ops')}")

    print(f"\nValidation: {valid} valid, {invalid} invalid out of {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="Build task pool from Ops-6K")
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(ROOT / "tasks" / "pool_v0.jsonl"))
    parser.add_argument("--validate", action="store_true", help="Validate existing pool")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.validate:
        validate_pool(output_path)
        return

    pool = build_task_pool(max_samples=args.max_samples, seed=args.seed)

    if not pool:
        print("ERROR: No evaluable tasks found!")
        sys.exit(1)

    save_pool(pool, output_path)

    # Also add existing local ops6k tasks from combined dataset
    combined_path = ROOT / "datasets" / "combined_kernelforge.jsonl"
    if combined_path.exists():
        local_count = 0
        with open(combined_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("evaluation_backend") == "ops6k" and row.get("task_code"):
                    code = row["task_code"]
                    if is_stateless_evaluable(code) and has_valid_structure(code):
                        local_count += 1
        print(f"\nAlso found {local_count} local ops6k tasks in combined dataset")


if __name__ == "__main__":
    main()
