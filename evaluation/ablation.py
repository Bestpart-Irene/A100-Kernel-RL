"""Ablation studies for KernelForge training pipeline.

Three hypotheses to validate:
  H1: Multi-stage RL improves over base model
  H2: RFT is necessary (without RFT -> collapse)
  H3: SKILL.md improves over generic prompts
"""
from __future__ import annotations

import os

from evaluation.eval_model import evaluate_checkpoint


def h1_multistage_improvement():
    """H1: Multi-stage RL improves over base model."""
    print("=== H1: Multi-Stage RL vs Base Model ===")

    base = evaluate_checkpoint("Qwen/Qwen2.5-Coder-7B-Instruct", num_problems=50)
    stage3_path = "outputs/kernelforge-stage3"

    if not os.path.exists(stage3_path):
        print("Stage 3 checkpoint not found. Train first.")
        return

    trained = evaluate_checkpoint(stage3_path, num_problems=50)

    print(f"\nBase:    compile={base['compile_rate']:.1%}, correct={base['correct_rate']:.1%}, reward={base['avg_reward']:.2f}")
    print(f"Trained: compile={trained['compile_rate']:.1%}, correct={trained['correct_rate']:.1%}, reward={trained['avg_reward']:.2f}")
    print(f"H1 {'CONFIRMED' if trained['avg_reward'] > base['avg_reward'] else 'NOT CONFIRMED'}: "
          f"reward delta = {trained['avg_reward'] - base['avg_reward']:+.2f}")


def h2_rft_necessity():
    """H2: RFT is necessary (without RFT -> training collapse)."""
    print("\n=== H2: RFT Necessity ===")

    stage1_path = "outputs/kernelforge-stage1"
    stage2_path = "outputs/kernelforge-stage2"
    stage3_path = "outputs/kernelforge-stage3"

    for path in [stage1_path, stage2_path, stage3_path]:
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}. Train all stages first.")
            return

    post_warmup = evaluate_checkpoint(stage1_path, num_problems=50)
    post_rft = evaluate_checkpoint(stage2_path, num_problems=50)
    final = evaluate_checkpoint(stage3_path, num_problems=50)

    print(f"\nPost warm-up (Stage 1): reward={post_warmup['avg_reward']:.2f}")
    print(f"Post RFT (Stage 2):     reward={post_rft['avg_reward']:.2f}")
    print(f"Final (Stage 3):        reward={final['avg_reward']:.2f}")
    print(f"H2: RFT contributes reward delta = {post_rft['avg_reward'] - post_warmup['avg_reward']:+.2f}")


def h3_skill_md_impact():
    """H3: SKILL.md improves over generic prompts."""
    print("\n=== H3: SKILL.md Impact ===")
    print("This ablation requires running Stage 3 with and without SKILL.md context.")
    print("Set KERNELFORGE_SKILL_FILE='' to disable SKILL.md, then re-evaluate.")
    print("Compare reward distributions between runs.")


def main():
    h1_multistage_improvement()
    h2_rft_necessity()
    h3_skill_md_impact()


if __name__ == "__main__":
    main()
