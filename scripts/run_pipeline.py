#!/usr/bin/env python3
"""KernelForge full training pipeline runner.

Usage:
    python scripts/run_pipeline.py              # Full pipeline
    python scripts/run_pipeline.py --dry-run    # Print plan without executing
    python scripts/run_pipeline.py --from-stage 2  # Resume from Stage 2
    python scripts/run_pipeline.py --eval-only  # Just run evaluation

Pipeline stages:
    0. Data preparation  (build combined dataset from manifest + Ops-6K)
    1. GRPO warm-up      (training/stage1_warmup.py)
    2. RFT + SFT         (training/stage2_rft.py)
    3. GRPO + curriculum  (training/stage3_grpo.py)
    4. Evaluation         (evaluation/compare_stages.py)

Training runs locally through the Python entrypoints. Remote evaluation is selected by
KERNELFORGE_EVAL_BACKEND (Modal by default, CoreWeave/Northflank optional).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


# --- Configuration ---

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
EVAL_BACKEND = os.getenv("KERNELFORGE_EVAL_BACKEND", "modal")
MODAL_APP = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
EVAL_URL = os.getenv("KERNELFORGE_EVAL_URL", "")

COMBINED_PATH = os.getenv("KERNELFORGE_COMBINED_PATH", "datasets/combined_kernelforge.jsonl")

STAGE1_OUTPUT = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
STAGE2_OUTPUT = os.getenv("KERNELFORGE_STAGE2_OUTPUT", "outputs/kernelforge-stage2")
STAGE3_OUTPUT = os.getenv("KERNELFORGE_STAGE3_OUTPUT", "outputs/kernelforge-stage3")


# --- Pipeline steps ---

STEPS = [
    {
        "name": "0. Build combined dataset (doubleGraph + Ops-6K)",
        "stage": 0,
        "check": lambda: os.path.exists(COMBINED_PATH),
        "cmd": [sys.executable, "datasets/build_combined_dataset.py"],
    },
    {
        "name": "1. GRPO warm-up (Stage 1)",
        "stage": 1,
        "check": lambda: os.path.isdir(STAGE1_OUTPUT),
        "cmd": [sys.executable, "-m", "training.stage1_warmup"],
    },
    {
        "name": "2. RFT + SFT (Stage 2)",
        "stage": 2,
        "check": lambda: os.path.isdir(STAGE2_OUTPUT),
        "cmd": [sys.executable, "-m", "training.stage2_rft"],
    },
    {
        "name": "3. GRPO + curriculum (Stage 3)",
        "stage": 3,
        "check": lambda: os.path.isdir(STAGE3_OUTPUT),
        "cmd": [sys.executable, "-m", "training.stage3_grpo"],
    },
    {
        "name": "4. Stage-over-stage evaluation",
        "stage": 4,
        "check": lambda: False,  # Always run evaluation
        "cmd": [sys.executable, "-m", "evaluation.compare_stages"],
    },
]


def _run_step(step: dict, dry_run: bool = False) -> bool:
    """Run a single pipeline step. Returns True on success."""
    name = step["name"]
    cached = step["check"]()

    if cached:
        print(f"  [{name}] CACHED — skipping")
        return True

    if dry_run:
        print(f"  [{name}] WOULD RUN: {' '.join(step['cmd'])}")
        return True

    print(f"\n{'='*60}")
    print(f"  [{name}] RUNNING...")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(step["cmd"], cwd=os.getcwd())

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"  [{name}] DONE ({elapsed:.1f}s)")
        return True
    else:
        print(f"  [{name}] FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        return False


def run_pipeline(from_stage: int = 0, dry_run: bool = False, eval_only: bool = False):
    """Execute the full training pipeline."""
    backend_label = (
        f"coreweave ({EVAL_URL})" if EVAL_BACKEND == "coreweave" and EVAL_URL else EVAL_BACKEND
    )
    if EVAL_BACKEND == "modal":
        backend_label = f"modal ({MODAL_APP})"
    print(f"KernelForge Pipeline — {TARGET_GPU} ({TARGET_ARCH}) via {backend_label}")
    print(f"{'DRY RUN' if dry_run else 'LIVE RUN'}")
    print()

    steps = STEPS
    if eval_only:
        steps = [s for s in STEPS if s["stage"] == 4]
    elif from_stage > 0:
        steps = [s for s in STEPS if s["stage"] >= from_stage]

    for step in steps:
        ok = _run_step(step, dry_run=dry_run)
        if not ok:
            print(f"\nPipeline stopped at: {step['name']}")
            print("Fix the issue and re-run with --from-stage to resume.")
            sys.exit(1)

    if not dry_run:
        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="KernelForge full training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without running")
    parser.add_argument("--from-stage", type=int, default=0, help="Resume from stage N (0=data, 1-3=training, 4=eval)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    run_pipeline(from_stage=args.from_stage, dry_run=args.dry_run, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
