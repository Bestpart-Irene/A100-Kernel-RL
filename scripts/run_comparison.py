#!/usr/bin/env python3
"""Head-to-head model comparison for KernelForge.

Compares two models on identical CUDA kernel tasks, with optional RL training.
Produces baseline (pre-RL) and post-RL benchmark results, then generates
4-way comparison reports using the existing compare_results infrastructure.

Usage:
    # Baseline-only comparison (no training)
    python scripts/run_comparison.py \
        --model-a Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --model-b Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled \
        --quant-bits 8 \
        --eval-only

    # Full comparison with RL training
    python scripts/run_comparison.py \
        --model-a Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --model-b Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled \
        --quant-bits 8 \
        --train-steps 50 \
        --output-dir results/coder_vs_opus

Phases:
    1. Baseline eval (both models, no RL)
    2. Train Model A (3-stage pipeline)
    3. Train Model B (3-stage pipeline)
    4. Post-RL eval (both models)
    5. Compare results (4-way delta reports)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run_benchmark(
    model_name: str,
    output_path: str,
    quant_bits: int = 0,
    max_tasks: int | None = None,
) -> dict:
    """Run benchmark for a model via subprocess (clean GPU state)."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
        "--model", model_name,
        "--output", output_path,
        "--quant-bits", str(quant_bits),
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])

    print(f"\n  Running benchmark: {model_name} ({quant_bits}bit) -> {output_path}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  Benchmark FAILED for {model_name} (exit {result.returncode})")
        return {}

    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def _run_training(
    model_name: str,
    quant_bits: int,
    output_prefix: str,
    train_steps: int = 50,
) -> str | None:
    """Run 3-stage training pipeline for a model. Returns stage3 output dir."""
    env = os.environ.copy()
    env["KERNELFORGE_MODEL"] = model_name
    env["KERNELFORGE_QUANT_BITS"] = str(quant_bits)
    env["KERNELFORGE_STAGE1_OUTPUT"] = f"{output_prefix}/stage1"
    env["KERNELFORGE_STAGE2_OUTPUT"] = f"{output_prefix}/stage2"
    env["KERNELFORGE_STAGE3_OUTPUT"] = f"{output_prefix}/stage3"
    env["KERNELFORGE_STAGE1_MAX_STEPS"] = str(train_steps)
    env["KERNELFORGE_STAGE3_MAX_STEPS"] = str(train_steps)

    stages = [
        ("Stage 1: GRPO Warmup", [sys.executable, "-m", "training.stage1_warmup"]),
        ("Stage 2: RFT/SFT", [sys.executable, "-m", "training.stage2_rft"]),
        ("Stage 3: GRPO + Curriculum", [sys.executable, "-m", "training.stage3_grpo"]),
    ]

    for stage_name, cmd in stages:
        print(f"\n  [{stage_name}] Training {model_name}...")
        start = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT), env=env)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  [{stage_name}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
            return None
        print(f"  [{stage_name}] DONE ({elapsed:.1f}s)")

    return f"{output_prefix}/stage3"


def _print_summary(comparisons: dict[str, dict]) -> None:
    """Print a final summary table of all comparisons."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    for label, comp in comparisons.items():
        if not comp:
            continue
        print(f"\n--- {label} ---")
        metrics = comp.get("metrics", {})
        for k in ("fast_0", "fast_1.05", "fast_compile", "geomean_vs_eager"):
            v = metrics.get(k, {})
            bv = v.get("before", 0)
            av = v.get("after", 0)
            dv = v.get("delta", 0)
            sign = "+" if dv > 0 else ""
            print(f"  {k:<20} {bv:.3f} -> {av:.3f} ({sign}{dv:.3f})")

        s = comp.get("summary", {})
        print(f"  Tasks: {s.get('improved', 0)} improved, "
              f"{s.get('regressed', 0)} regressed, "
              f"{s.get('unchanged', 0)} unchanged")

    print("\n" + "=" * 70)


def run_comparison(
    model_a: str,
    model_b: str,
    quant_bits: int = 8,
    output_dir: str = "results/comparison",
    max_eval_tasks: int | None = None,
    train_steps: int = 50,
    eval_only: bool = False,
    label_a: str = "model_a",
    label_b: str = "model_b",
):
    """Run full head-to-head comparison."""
    from scripts.compare_results import compare, print_comparison

    os.makedirs(output_dir, exist_ok=True)
    quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(quant_bits, f"{quant_bits}bit")

    print("=" * 70)
    print("KERNELFORGE HEAD-TO-HEAD MODEL COMPARISON")
    print("=" * 70)
    print(f"  Model A ({label_a}): {model_a}")
    print(f"  Model B ({label_b}): {model_b}")
    print(f"  Quantization: {quant_label}")
    print(f"  Output: {output_dir}")
    if not eval_only:
        print(f"  Train steps: {train_steps}")
    print()

    # Phase 1: Baseline evaluation
    print("\n### Phase 1: Baseline Evaluation (no RL) ###")
    baseline_a = _run_benchmark(
        model_a, f"{output_dir}/{label_a}_baseline.json",
        quant_bits=quant_bits, max_tasks=max_eval_tasks,
    )
    baseline_b = _run_benchmark(
        model_b, f"{output_dir}/{label_b}_baseline.json",
        quant_bits=quant_bits, max_tasks=max_eval_tasks,
    )

    comparisons: dict[str, dict] = {}

    # Head-to-head baseline
    if baseline_a and baseline_b:
        comp = compare(baseline_a, baseline_b)
        comparisons["Baseline: Model A vs Model B"] = comp
        print_comparison(comp)

    if eval_only:
        _print_summary(comparisons)
        _save_summary(comparisons, output_dir)
        return comparisons

    # Phase 2: Train Model A
    print(f"\n### Phase 2: Train Model A ({label_a}) ###")
    ckpt_a = _run_training(
        model_a, quant_bits,
        output_prefix=f"{output_dir}/training/{label_a}",
        train_steps=train_steps,
    )

    # Phase 3: Train Model B
    print(f"\n### Phase 3: Train Model B ({label_b}) ###")
    ckpt_b = _run_training(
        model_b, quant_bits,
        output_prefix=f"{output_dir}/training/{label_b}",
        train_steps=train_steps,
    )

    # Phase 4: Post-RL evaluation
    print("\n### Phase 4: Post-RL Evaluation ###")
    post_rl_a = {}
    post_rl_b = {}
    if ckpt_a:
        post_rl_a = _run_benchmark(
            ckpt_a, f"{output_dir}/{label_a}_post_rl.json",
            quant_bits=quant_bits, max_tasks=max_eval_tasks,
        )
    if ckpt_b:
        post_rl_b = _run_benchmark(
            ckpt_b, f"{output_dir}/{label_b}_post_rl.json",
            quant_bits=quant_bits, max_tasks=max_eval_tasks,
        )

    # Phase 5: Compare
    print("\n### Phase 5: Comparisons ###")

    if baseline_a and post_rl_a:
        comp = compare(baseline_a, post_rl_a)
        comparisons[f"RL Gain: {label_a}"] = comp
        print_comparison(comp)

    if baseline_b and post_rl_b:
        comp = compare(baseline_b, post_rl_b)
        comparisons[f"RL Gain: {label_b}"] = comp
        print_comparison(comp)

    if post_rl_a and post_rl_b:
        comp = compare(post_rl_a, post_rl_b)
        comparisons["Post-RL: Model A vs Model B"] = comp
        print_comparison(comp)

    _print_summary(comparisons)
    _save_summary(comparisons, output_dir)
    return comparisons


def _save_summary(comparisons: dict[str, dict], output_dir: str) -> None:
    """Save all comparisons to a summary JSON."""
    summary_path = f"{output_dir}/comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull comparison saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head model comparison for CUDA kernel generation"
    )
    parser.add_argument("--model-a", required=True, help="First model (HF model ID or checkpoint)")
    parser.add_argument("--model-b", required=True, help="Second model (HF model ID or checkpoint)")
    parser.add_argument("--label-a", default="coder_moe", help="Short label for model A")
    parser.add_argument("--label-b", default="opus_moe", help="Short label for model B")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[0, 4, 8],
                        help="Quantization: 0=bf16, 4=NF4, 8=INT8 (default: 8)")
    parser.add_argument("--output-dir", default="results/comparison",
                        help="Directory for all outputs")
    parser.add_argument("--max-eval-tasks", type=int, default=None,
                        help="Limit eval to N tasks (faster iteration)")
    parser.add_argument("--train-steps", type=int, default=50,
                        help="Training steps per stage (default: 50)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only compare baseline generation quality")
    args = parser.parse_args()

    run_comparison(
        model_a=args.model_a,
        model_b=args.model_b,
        quant_bits=args.quant_bits,
        output_dir=args.output_dir,
        max_eval_tasks=args.max_eval_tasks,
        train_steps=args.train_steps,
        eval_only=args.eval_only,
        label_a=args.label_a,
        label_b=args.label_b,
    )


if __name__ == "__main__":
    main()
