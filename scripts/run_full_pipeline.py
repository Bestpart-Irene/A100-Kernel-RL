#!/usr/bin/env python3
"""Compatibility wrapper for the current KernelForge training pipeline.

The active path lives in:
  - training.grpo_train        for preflight + Stage 1/2/3 execution
  - evaluation.compare_stages  for stage-over-stage evaluation

This wrapper preserves older commands while routing them to the supported
entrypoints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], description: str) -> int:
    print(f"\n{'=' * 60}")
    print(description)
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def stage_command(stage: str) -> tuple[list[str], str]:
    stage_key = stage.lower()
    if stage_key == "preflight":
        return ([sys.executable, "-m", "training.grpo_train", "--preflight-only"], "KernelForge preflight")
    if stage_key == "stage1":
        return ([sys.executable, "-m", "training.grpo_train", "--stage", "stage1"], "KernelForge Stage 1")
    if stage_key in {"stage2", "sft"}:
        return ([sys.executable, "-m", "training.grpo_train", "--stage", "stage2"], "KernelForge Stage 2")
    if stage_key in {"stage3", "grpo"}:
        return ([sys.executable, "-m", "training.grpo_train", "--stage", "stage3"], "KernelForge Stage 3")
    if stage_key in {"pipeline", "all"}:
        return ([sys.executable, "-m", "training.grpo_train", "--stage", "pipeline"], "KernelForge full pipeline")
    if stage_key in {"eval", "test"}:
        return ([sys.executable, "-m", "evaluation.compare_stages"], "KernelForge evaluation")
    raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description="KernelForge compatibility pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["preflight", "stage1", "stage2", "stage3", "pipeline", "eval", "all", "sft", "grpo", "test"],
        default="pipeline",
        help="Supported pipeline stage or legacy alias",
    )
    args = parser.parse_args()

    cmd, description = stage_command(args.stage)
    return run_command(cmd, description)

if __name__ == "__main__":
    raise SystemExit(main())
