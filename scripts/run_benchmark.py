#!/usr/bin/env python3
"""Benchmark runner for KernelForge — measures fast_p metrics on A100.

Runs a model against a set of tasks, evaluates each via the eval backend,
and reports KernelBench-compatible metrics (fast_0, fast_1.05, fast_compile).

Usage:
    # Compute baselines only (no model needed)
    python scripts/run_benchmark.py --baselines-only

    # Run benchmark with a model
    python scripts/run_benchmark.py --model unsloth/Qwen3-Coder-30B-A3B-Instruct

    # Run on specific task pool
    python scripts/run_benchmark.py --pool tasks/pool_v0.jsonl --output results/run1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compute_fast_p(results: list[dict], p: float = 1.0) -> float:
    """Compute fast_p: fraction of tasks that are correct AND speedup >= p."""
    if not results:
        return 0.0
    count = sum(
        1 for r in results
        if r.get("correct") and r.get("speedup_vs_eager", 0) >= p
    )
    return count / len(results)


def compute_fast_compile(results: list[dict], threshold: float = 1.05) -> float:
    """Fraction of tasks correct AND faster than torch.compile by threshold."""
    if not results:
        return 0.0
    count = sum(
        1 for r in results
        if r.get("correct") and r.get("speedup_vs_compile", 0) >= threshold
    )
    return count / len(results)


def compute_geomean_speedup(results: list[dict], key: str = "speedup_vs_eager") -> float:
    """Geometric mean speedup across correct tasks."""
    speedups = [
        r[key] for r in results
        if r.get("correct") and r.get(key, 0) > 0
    ]
    if not speedups:
        return 0.0
    log_sum = sum(math.log(s) for s in speedups)
    return math.exp(log_sum / len(speedups))


def run_benchmark(
    pool_path: str | None = None,
    model_name: str | None = None,
    output_path: str | None = None,
    baselines_only: bool = False,
    max_tasks: int | None = None,
    quant_bits: int = 0,
    benchmark: str = "all",
):
    """Run the benchmark and output results.

    Args:
        benchmark: Filter tasks by evaluation_backend. Options:
            "all" — run all evaluable tasks (default)
            "ops6k" — CUDA Agent Ops-6K tasks only
            "wcc" — Dr. Kernel WCC graph kernels only
    """
    from openenv_env.task_pool import TaskPool
    from training.task_support import normalize_task_row

    pool = TaskPool.load(pool_path)
    print(f"Loaded task pool: {pool.summary()}")

    tasks = pool.tasks

    # Filter by benchmark suite
    if benchmark != "all":
        tasks = [t for t in tasks if t.get("evaluation_backend") == benchmark]
        print(f"Filtered to benchmark={benchmark}: {len(tasks)} tasks")
    else:
        # Exclude unsupported tasks (no evaluator)
        tasks = [t for t in tasks if t.get("evaluation_backend", "unsupported") != "unsupported"]
        print(f"Using all evaluable tasks: {len(tasks)} tasks")

    if max_tasks:
        tasks = tasks[:max_tasks]

    results = []
    for i, task in enumerate(tasks):
        task = normalize_task_row(task)
        tid = task.get("task_id", f"task_{i}")
        backend = task.get("evaluation_backend", "unknown")
        ops = task.get("ops", [])

        print(f"\n[{i+1}/{len(tasks)}] {tid} ({backend}) ops={ops}")

        if baselines_only:
            # Just report task info, no model evaluation
            results.append({
                "task_id": tid,
                "backend": backend,
                "ops": ops,
                "correct": None,
                "speedup_vs_eager": None,
                "speedup_vs_compile": None,
                "status": "baselines_only",
            })
            continue

        if not model_name:
            print("  SKIP: no model specified")
            results.append({
                "task_id": tid,
                "backend": backend,
                "ops": ops,
                "correct": False,
                "speedup_vs_eager": 0.0,
                "speedup_vs_compile": 0.0,
                "status": "no_model",
            })
            continue

        # Generate CUDA code using the model
        try:
            cuda_code = _generate_kernel(task, model_name, quant_bits=quant_bits)
        except Exception as exc:
            print(f"  GENERATION FAILED: {exc}")
            results.append({
                "task_id": tid,
                "backend": backend,
                "ops": ops,
                "correct": False,
                "speedup_vs_eager": 0.0,
                "speedup_vs_compile": 0.0,
                "status": "generation_failed",
                "error": str(exc)[:200],
            })
            continue

        # Evaluate via the eval backend
        try:
            from training.task_support import evaluate_code_remote

            eval_result = evaluate_code_remote(cuda_code, task)
            correct = bool(eval_result.get("correct"))
            su_eager = float(eval_result.get("speedup_vs_orig", 0) or 0)
            su_compile = float(eval_result.get("speedup_vs_dg", 0) or 0)
            reward = float(eval_result.get("reward", -1))

            print(f"  correct={correct} su_eager={su_eager:.2f}x "
                  f"su_compile={su_compile:.2f}x reward={reward}")

            results.append({
                "task_id": tid,
                "backend": backend,
                "ops": ops,
                "correct": correct,
                "speedup_vs_eager": su_eager,
                "speedup_vs_compile": su_compile,
                "reward": reward,
                "runtime_ms": eval_result.get("runtime_ms", 0),
                "baseline_eager_ms": eval_result.get("baseline_eager_ms", 0),
                "baseline_compile_ms": eval_result.get("baseline_compile_ms", 0),
                "status": "evaluated",
            })
        except Exception as exc:
            print(f"  EVAL FAILED: {exc}")
            results.append({
                "task_id": tid,
                "backend": backend,
                "ops": ops,
                "correct": False,
                "speedup_vs_eager": 0.0,
                "speedup_vs_compile": 0.0,
                "status": "eval_failed",
                "error": str(exc)[:200],
            })

    # Compute metrics
    evaluated = [r for r in results if r.get("status") == "evaluated"]
    metrics = {
        "total_tasks": len(tasks),
        "evaluated": len(evaluated),
        "fast_0": compute_fast_p(evaluated, p=0.0),
        "fast_1": compute_fast_p(evaluated, p=1.0),
        "fast_1.05": compute_fast_p(evaluated, p=1.05),
        "fast_compile": compute_fast_compile(evaluated),
        "geomean_vs_eager": compute_geomean_speedup(evaluated, "speedup_vs_eager"),
        "geomean_vs_compile": compute_geomean_speedup(evaluated, "speedup_vs_compile"),
        "compile_rate": (
            sum(1 for r in evaluated if r.get("correct") is not None) / len(evaluated)
            if evaluated else 0.0
        ),
        "correctness_rate": compute_fast_p(evaluated, p=0.0),
    }

    output = {
        "model": model_name or "none",
        "pool_path": str(pool_path or "default"),
        "benchmark": benchmark,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
        "results": results,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)

    # Save results
    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return output


def _generate_kernel(task: dict, model_name: str, quant_bits: int = 0) -> str:
    """Generate a CUDA kernel for a task using the specified model.

    This is a placeholder — in production, this calls the model via
    vLLM, Unsloth, or the TRL rollout infrastructure.
    """
    from openenv_env.skill_builder import build_skill_md
    from training.task_support import build_generation_prompt

    prompt = build_generation_prompt(
        task,
        skill_context=build_skill_md("a100"),
    )

    # Try to use vllm or transformers for generation
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quant_bits in (4, 8):
            from training.model_loader import _make_bnb_config
            bnb_config = _make_bnb_config(quant_bits)
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract CUDA code from response
        if "```" in response:
            blocks = response.split("```")
            for i, block in enumerate(blocks):
                if i % 2 == 1:  # Inside code block
                    # Strip language identifier
                    lines = block.strip().split("\n")
                    if lines and lines[0].strip().lower() in ("cuda", "cpp", "c++", "c"):
                        lines = lines[1:]
                    return "\n".join(lines)

        return response

    except ImportError:
        raise RuntimeError(
            f"Cannot load model {model_name}. "
            "Install transformers: pip install transformers torch"
        )


def main():
    parser = argparse.ArgumentParser(description="KernelForge Benchmark Runner")
    parser.add_argument("--pool", type=str, default=None, help="Task pool JSONL path")
    parser.add_argument("--model", type=str, default=None, help="Model name/path")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--quant-bits", type=int, default=0, choices=[0, 4, 8],
                        help="Quantization bits: 0=bf16, 4=NF4, 8=INT8 (recommended)")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "ops6k", "wcc"],
                        help="Benchmark suite: all, ops6k (CUDA Agent), wcc (Dr. Kernel)")
    args = parser.parse_args()

    run_benchmark(
        pool_path=args.pool,
        model_name=args.model,
        output_path=args.output,
        baselines_only=args.baselines_only,
        max_tasks=args.max_tasks,
        quant_bits=args.quant_bits,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
