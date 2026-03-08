"""Evaluate a trained model on held-out, evaluator-backed tasks."""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from training.multi_turn_rollout import extract_cuda_code
from training.task_support import (
    build_generation_prompt,
    evaluate_code_remote,
    filter_supported_tasks,
    summarize_tasks,
)
TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
EVAL_PROBLEMS = 50


def evaluate_checkpoint(
    checkpoint_path: str,
    num_problems: int = EVAL_PROBLEMS,
    temperature: float = 0.7,
) -> dict:
    """Evaluate a model checkpoint on held-out supported tasks."""
    from transformers import pipeline

    from training.model_loader import load_model_and_tokenizer

    print(f"Evaluating {checkpoint_path} on {num_problems} problems...")

    model, tokenizer = load_model_and_tokenizer(checkpoint_path=checkpoint_path)
    tasks = _load_eval_tasks(num_problems)

    results = {
        "checkpoint": checkpoint_path,
        "num_problems": len(tasks),
        "compiles": 0,
        "correct": 0,
        "rewards": [],
        "speedups": [],
        "skipped": 0,
    }

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for idx, task in enumerate(tasks):
        prompt = build_generation_prompt(task)
        try:
            output = generator(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=temperature,
            )
            text = output[0]["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt) :]

            code = extract_cuda_code(text)
            if not code:
                raise RuntimeError("Model response did not contain CUDA/C++ code")

            result = evaluate_code_remote(code, task)
            reward = float(result.get("reward", -1.0))

            if result.get("compiles"):
                results["compiles"] += 1
            if result.get("correct"):
                results["correct"] += 1
            results["rewards"].append(reward)
            if result.get("correct") and result.get("speedup_vs_orig", 0) > 0:
                results["speedups"].append(float(result["speedup_vs_orig"]))

        except Exception as exc:
            print(f"  Problem {idx + 1} failed: {exc}")
            results["rewards"].append(-1.0)

        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{len(tasks)}")

    n = results["num_problems"]
    results["compile_rate"] = results["compiles"] / n if n > 0 else 0
    results["correct_rate"] = results["correct"] / n if n > 0 else 0
    results["avg_reward"] = statistics.mean(results["rewards"]) if results["rewards"] else 0
    results["median_speedup"] = (
        statistics.median(results["speedups"]) if results["speedups"] else 0
    )
    return results


def _load_eval_tasks(num_problems: int) -> list[dict]:
    """Load held-out evaluator-backed tasks from the combined dataset."""
    combined_path = Path("datasets/combined_kernelforge.jsonl")
    if combined_path.exists():
        with combined_path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        supported = filter_supported_tasks(rows)
        if supported:
            return supported[-num_problems:]

    print(
        "WARNING: no supported combined dataset tasks found. "
        "Falling back to a minimal built-in WCC evaluation set."
    )
    fallback = [
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using union-find with path compression."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "eval_fallback",
        }
        for _ in range(num_problems)
    ]
    return fallback


def evaluate_multi_seed(
    checkpoint_path: str,
    num_problems: int = EVAL_PROBLEMS,
    num_seeds: int = 3,
    temperatures: tuple[float, ...] = (0.2, 0.7, 1.0),
) -> dict:
    """Run evaluation across multiple seeds and temperatures."""
    all_results = []
    for seed in range(num_seeds):
        for temp in temperatures:
            print(f"\n--- Seed {seed}, temp {temp} ---")
            result = evaluate_checkpoint(
                checkpoint_path,
                num_problems=num_problems,
                temperature=temp,
            )
            result["seed"] = seed
            result["temperature"] = temp
            all_results.append(result)

    compile_rates = [r["compile_rate"] for r in all_results]
    correct_rates = [r["correct_rate"] for r in all_results]
    avg_rewards = [r["avg_reward"] for r in all_results]

    return {
        "checkpoint": checkpoint_path,
        "num_runs": len(all_results),
        "compile_rate": _summarize_metric(compile_rates),
        "correct_rate": _summarize_metric(correct_rates),
        "avg_reward": _summarize_metric(avg_rewards),
        "dataset_summary": summarize_tasks(_load_eval_tasks(num_problems)),
        "raw_results": all_results,
    }


def _ci_95(values: list[float]) -> tuple[float, float]:
    """95% CI using a simple t-value approximation."""
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0.0
        return (mean, mean)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / (n ** 0.5)
    t_val = 2.776 if n <= 5 else 2.228 if n <= 10 else 1.96
    return (mean - t_val * se, mean + t_val * se)


def _summarize_metric(values: list[float]) -> dict[str, float | tuple[float, float]]:
    """Summarize a metric across multiple runs."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_95": (0.0, 0.0)}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci_95": _ci_95(values),
    }


def compare_stages():
    """Compare all training stages using the supported eval set."""
    stages = [
        ("Base", None),
        ("Stage 1 (GRPO warm-up)", "outputs/kernelforge-stage1"),
        ("Stage 2 (RFT)", "outputs/kernelforge-stage2"),
        ("Stage 3 (GRPO+curriculum)", "outputs/kernelforge-stage3"),
    ]

    print("=== Model Evaluation Across Training Stages ===\n")

    for name, path in stages:
        if path and not os.path.exists(path):
            print(f"  {name}: checkpoint not found at {path}, skipping")
            continue

        model_path = path or os.getenv("KERNELFORGE_FALLBACK_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        results = evaluate_checkpoint(model_path)
        print(f"\n{name}:")
        print(f"  Compile rate: {results['compile_rate']:.1%}")
        print(f"  Correct rate: {results['correct_rate']:.1%}")
        print(f"  Avg reward:   {results['avg_reward']:.2f}")
        print(f"  Median speedup: {results['median_speedup']:.2f}x")


def main():
    compare_stages()


if __name__ == "__main__":
    main()
