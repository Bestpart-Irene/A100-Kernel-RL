"""Compare model quality across all training stages.

Evaluates: Base → Stage 1 (GRPO warm-up) → Stage 2 (RFT) → Stage 3 (GRPO+curriculum)
Reports: compile rate, correctness rate, avg reward, speedup distribution per stage.
"""
from __future__ import annotations

import os

from evaluation.eval_model import evaluate_checkpoint


STAGES = [
    ("Base", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("Stage 1 (GRPO warm-up)", "outputs/kernelforge-stage1"),
    ("Stage 2 (RFT)", "outputs/kernelforge-stage2"),
    ("Stage 3 (GRPO+curriculum)", "outputs/kernelforge-stage3"),
]


def compare_all_stages(num_problems: int = 50) -> list[dict]:
    """Evaluate all stages and print comparison table."""
    print("=" * 70)
    print("STAGE-OVER-STAGE COMPARISON")
    print("=" * 70)

    results = []
    for stage_name, checkpoint_path in STAGES:
        if not os.path.exists(checkpoint_path) and "/" not in checkpoint_path:
            print(f"\n--- {stage_name}: SKIPPED (checkpoint not found: {checkpoint_path}) ---")
            results.append({"stage": stage_name, "path": checkpoint_path, "skipped": True})
            continue

        print(f"\n--- Evaluating: {stage_name} ---")
        try:
            metrics = evaluate_checkpoint(checkpoint_path, num_problems=num_problems)
            metrics["stage"] = stage_name
            metrics["path"] = checkpoint_path
            results.append(metrics)

            print(f"  Compile rate:    {metrics['compile_rate']:.1%}")
            print(f"  Correct rate:    {metrics['correct_rate']:.1%}")
            print(f"  Avg reward:      {metrics['avg_reward']:.2f}")
            print(f"  Median speedup:  {metrics.get('median_speedup', 0):.2f}x")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"stage": stage_name, "path": checkpoint_path, "error": str(e)})

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Stage':<30} {'Compile':>8} {'Correct':>8} {'Reward':>8} {'Speedup':>8}")
    print("-" * 70)
    for r in results:
        if r.get("skipped"):
            print(f"{r['stage']:<30} {'SKIPPED':>8}")
        elif r.get("error"):
            print(f"{r['stage']:<30} {'ERROR':>8}")
        else:
            print(
                f"{r['stage']:<30} "
                f"{r['compile_rate']:>7.1%} "
                f"{r['correct_rate']:>7.1%} "
                f"{r['avg_reward']:>7.2f} "
                f"{r.get('median_speedup', 0):>7.2f}x"
            )
    print("=" * 70)

    # Compute deltas
    valid = [r for r in results if not r.get("skipped") and not r.get("error")]
    if len(valid) >= 2:
        base = valid[0]
        final = valid[-1]
        delta = final["avg_reward"] - base["avg_reward"]
        print(f"\nReward delta (base → final): {delta:+.2f}")
        print(f"Pipeline {'IMPROVED' if delta > 0 else 'DID NOT IMPROVE'} model quality.")

    return results


def main():
    compare_all_stages()


if __name__ == "__main__":
    main()
