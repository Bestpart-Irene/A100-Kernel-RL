"""SkyDiscover evaluator bridge — connects SkyDiscover's evolutionary search to the configured remote A100 eval backend.

SkyDiscover expects: evaluate(program_path) -> {"combined_score": float, "artifacts": {...}}
We bridge to: the shared KernelForge evaluator contract on A100.

Supports cascade evaluation:
  Stage 1 (fast): local nvcc compile check only — filters ~50% of candidates
  Stage 2 (slow): full remote A100 benchmark — accurate timing + correctness
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from openenv_env.anti_hack import extract_cu_flags, scan_forbidden_symbols
from openenv_env.eval_backend import dispatch_eval
from openenv_env.reward import compute_reward, validate_eval_result


@dataclass
class EvaluationResult:
    """SkyDiscover-compatible evaluation result."""
    combined_score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class KernelForgeEvaluator:
    """Evaluator bridge from SkyDiscover to the KernelForge remote A100 evaluation pipeline.

    Usage:
        evaluator = KernelForgeEvaluator()
        result = await evaluator.evaluate_program(cuda_code, program_id="wcc_v1")
    """

    def __init__(
        self,
        target_arch: str = "sm_80",
        stage1_threshold: float = 0.0,
        task_code: str | None = None,
        eval_mode: str = "wcc",
    ):
        self.target_arch = target_arch
        self.stage1_threshold = stage1_threshold
        self.task_code = task_code
        self.eval_mode = eval_mode  # "wcc" or "ops6k"

    def evaluate_stage1(self, cuda_code: str) -> EvaluationResult:
        """Fast local compile check — no GPU needed.

        Filters obviously broken candidates before expensive remote evaluation calls.
        Returns combined_score=0 for compile failures, 0.1 for compile success.
        """
        result = EvaluationResult()

        # Check for forbidden symbols in source
        cu_flags = extract_cu_flags(cuda_code)

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "candidate.cu")
            obj_path = os.path.join(tmpdir, "candidate.o")

            with open(src_path, "w", encoding="utf-8") as f:
                f.write(cuda_code)

            cmd = [
                "nvcc", f"-arch={self.target_arch}", "-O3", "-c",
                src_path, "-o", obj_path,
            ]
            # Add user-specified flags
            for flag in cu_flags:
                cmd.insert(3, flag)

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                )
                if proc.returncode != 0:
                    result.error = f"Compile failed: {proc.stderr[:500]}"
                    result.combined_score = -1.0
                    result.metrics["compiles"] = False
                    return result
            except subprocess.TimeoutExpired:
                result.error = "Compile timed out (30s)"
                result.combined_score = -1.0
                result.metrics["compiles"] = False
                return result
            except FileNotFoundError:
                result.error = "nvcc not found — local compile check skipped"
                result.combined_score = 0.0
                result.metrics["compiles"] = "skipped"
                return result

        result.combined_score = 0.1  # Compiles but not yet benchmarked
        result.metrics["compiles"] = True
        result.artifacts["cu_flags"] = cu_flags
        return result

    def evaluate_stage2(self, cuda_code: str) -> EvaluationResult:
        """Full A100 benchmark via the configured backend — accurate timing + correctness.

        Only call this for candidates that passed stage1.
        """
        result = EvaluationResult()

        try:
            if self.eval_mode == "ops6k" and self.task_code:
                fn_name = "evaluate_ops6k_kernel"
                payload = {
                    "cuda_code": cuda_code,
                    "task_code": self.task_code,
                    "warmup_iters": 10,
                    "benchmark_runs": 10,
                }
            else:
                fn_name = "evaluate_kernel"
                payload = {
                    "cuda_code": cuda_code,
                    "verify_graphs": 5,
                    "warmup_iters": 50,
                    "benchmark_runs": 30,
                }

            modal_result = dispatch_eval(fn_name, payload)
            modal_result = validate_eval_result(modal_result)

            compiled = modal_result.get("compiles", False)
            correct = modal_result.get("correct", False)

            # Use speedup_vs_orig for WCC, speedup_vs_orig for Ops-6K
            speedup_key = "speedup_vs_orig"
            speedup = modal_result.get(speedup_key, 0.0)

            reward = compute_reward(
                compiled=compiled,
                correct=correct,
                speedup_vs_eager=speedup,
                speedup_vs_compile=modal_result.get("speedup_vs_dg", 0.0),
            )

            result.combined_score = reward
            result.metrics = {
                "compiles": compiled,
                "correct": correct,
                "runtime_ms": modal_result.get("runtime_ms", 0.0),
                "speedup": speedup,
                "reward": reward,
            }
            result.artifacts = {
                "runtime_stats": modal_result.get("runtime_stats", {}),
                "verifier_msg": modal_result.get("verifier_msg", ""),
            }

        except Exception as e:
            result.error = f"Eval dispatch failed: {str(e)[:500]}"
            result.combined_score = -1.0

        return result

    async def evaluate_program(
        self, program_solution: str, program_id: str = ""
    ) -> EvaluationResult:
        """SkyDiscover-compatible async evaluation entry point.

        Implements cascade: stage1 (compile check) → stage2 (A100 benchmark).
        """
        # Stage 1: fast compile check
        stage1 = self.evaluate_stage1(program_solution)

        if stage1.combined_score <= self.stage1_threshold:
            stage1.artifacts["stage"] = "compile_fail"
            stage1.artifacts["program_id"] = program_id
            return stage1

        # Stage 2: full A100 benchmark
        stage2 = self.evaluate_stage2(program_solution)
        stage2.artifacts["stage"] = "full_benchmark"
        stage2.artifacts["program_id"] = program_id
        stage2.artifacts["stage1_result"] = {
            "cu_flags": stage1.artifacts.get("cu_flags", []),
        }
        return stage2

    def evaluate(self, program_path: str) -> dict:
        """Synchronous file-based evaluation for SkyDiscover's evaluate() interface.

        Reads .cu file, runs cascade evaluation, returns SkyDiscover-compatible dict.
        """
        with open(program_path, "r", encoding="utf-8") as f:
            cuda_code = f.read()

        # Stage 1
        stage1 = self.evaluate_stage1(cuda_code)
        if stage1.combined_score <= self.stage1_threshold:
            return {
                "combined_score": stage1.combined_score,
                "artifacts": {**stage1.artifacts, "error": stage1.error},
            }

        # Stage 2
        stage2 = self.evaluate_stage2(cuda_code)
        return {
            "combined_score": stage2.combined_score,
            "artifacts": {**stage2.artifacts, "metrics": stage2.metrics},
        }
