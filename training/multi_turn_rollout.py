"""
Multi-turn rollout for GRPOTrainer.

The policy generates on the training GPU, while correctness and runtime reward
are computed remotely on the target A100 via CoreWeave/Northflank (or Modal).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from training.curriculum import format_topology_context
from training.run_metadata import utc_timestamp_rfc3339
from training.task_support import (
    build_generation_prompt,
    build_prompt_lookup,
    compute_task_reward,
    evaluate_code_remote,
    normalize_eval_result,
    normalize_task_row,
)
LOCAL_COMPILE_CHECK = os.getenv("KERNELFORGE_LOCAL_COMPILE", "1") == "1"
TARGET_CUDA_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
MAX_FEEDBACK_CHARS = int(os.getenv("KERNELFORGE_MAX_FEEDBACK_CHARS", "1200"))
MAX_ERROR_CHARS = int(os.getenv("KERNELFORGE_MAX_ERROR_CHARS", "800"))
ROLLOUT_LOG_PATH = Path(
    os.getenv("KERNELFORGE_ROLLOUT_LOG", "outputs/rollout_metrics.jsonl")
).resolve()


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from model output (fenced block or raw __global__)."""
    for marker in ["```cuda", "```cpp", "```c", "```c++"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

    if re.search(r"__global__\s+void\s+\w+", text) or "PYBIND11_MODULE" in text:
        return text.strip()
    return ""


def _append_rollout_log(record: dict[str, Any]) -> None:
    """Append one rollout event to a JSONL log file."""
    try:
        ROLLOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload.setdefault("timestamp", utc_timestamp_rfc3339())
        with ROLLOUT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def _local_compile_check(code: str) -> tuple[bool, str]:
    """Quick local nvcc syntax check before paying for a remote evaluation."""
    if not LOCAL_COMPILE_CHECK:
        return True, ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(code)
            cu_path = f.name

        obj_path = cu_path.replace(".cu", ".o")
        proc = subprocess.run(
            ["nvcc", f"-arch={TARGET_CUDA_ARCH}", "-c", cu_path, "-o", obj_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        for path in (cu_path, obj_path):
            try:
                os.unlink(path)
            except OSError:
                pass

        if proc.returncode != 0:
            return False, proc.stderr[:1000]
        return True, ""
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Local compile timed out (15s)"
    except Exception:
        return True, ""


def _compute_reward_from_result(result: dict) -> float:
    """Compute discrete milestone reward from evaluation result."""
    return compute_task_reward(result)


def _format_feedback(result: dict, reward: float, turn: int) -> str:
    """Format evaluator feedback for the next policy turn."""
    result = normalize_eval_result(result)
    parts = [f"[Turn {turn + 1} Result]"]

    if not result.get("compiles"):
        error = result.get("error", "unknown compilation error")
        parts.append(f"COMPILATION FAILED:\n{error[:MAX_ERROR_CHARS]}")
        parts.append("Fix the compilation errors above and resubmit.")
    elif not result.get("correct"):
        msg = result.get("verifier_msg") or result.get("error") or "unknown verification failure"
        parts.append(f"VERIFICATION FAILED: {msg}")
        parts.append("Your kernel produced incorrect output. Fix the implementation.")
    else:
        runtime = float(result.get("runtime_ms", 0.0) or 0.0)
        speedup_eager = float(result.get("speedup_vs_orig", 0.0) or 0.0)
        speedup_compile = float(result.get("speedup_vs_dg", 0.0) or 0.0)
        stats = result.get("runtime_stats", {}) or {}
        parts.append(f"CORRECT. Runtime: {runtime:.3f}ms")
        parts.append(f"  Speedup vs eager: {speedup_eager:.2f}x")
        if speedup_compile:
            parts.append(f"  Speedup vs torch.compile: {speedup_compile:.2f}x")
        if stats:
            parts.append(
                f"  Stats: mean={float(stats.get('mean', 0.0)):.3f}ms, "
                f"std={float(stats.get('std', 0.0)):.3f}ms"
            )

        if reward <= 1.0:
            parts.append(
                "Kernel is correct but not faster than eager PyTorch. "
                "Try reducing memory traffic or using shared memory tiling."
            )
        elif reward <= 2.0:
            parts.append(
                "Faster than eager PyTorch but not torch.compile. Push toward "
                "beating torch.compile with better occupancy or warp-level primitives."
            )

    feedback = "\n".join(parts)
    return feedback[:MAX_FEEDBACK_CHARS]


_baselines_cache: dict[str, Any] | None = None


def _get_baselines() -> tuple[float | None, float | None]:
    """Fetch baseline timings from eval backend (cached across calls)."""
    global _baselines_cache
    if _baselines_cache is None:
        try:
            from openenv_env.eval_backend import dispatch_eval

            _baselines_cache = dispatch_eval("profile_baselines") or {}
        except Exception as exc:
            print(f"Baseline profiling failed: {exc}")
            _baselines_cache = {}
    return _baselines_cache.get("original_ms"), _baselines_cache.get("doublegraph_ms")


def make_multi_turn_rollout(
    max_turns: int = 3,
    skill_md_gpu: str | None = None,
    problem_metadata: list[dict] | None = None,
) -> Callable:
    """Create a task-aware rollout_func for GRPOTrainer."""
    from trl.experimental.openenv import generate_rollout_completions
    from openenv_env.skill_builder import build_skill_md

    gpu_name = skill_md_gpu or os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()
    prompt_lookup = build_prompt_lookup(problem_metadata or [])

    def rollout_func(prompts: list[str], trainer: Any) -> dict:
        tokenizer = trainer.processing_class
        skill_context = build_skill_md(gpu_name)
        baseline_orig, baseline_dg = _get_baselines()

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_best_rewards: list[float] = []

        for prompt_idx, prompt in enumerate(prompts):
            task_row = normalize_task_row(prompt_lookup.get(prompt, {"prompt": prompt}))
            topology_ctx = format_topology_context(task_row)
            current_prompt = build_generation_prompt(
                task_row,
                skill_context=skill_context,
                topology_context=topology_ctx,
            )

            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            best_reward = -1.0

            for turn in range(max_turns):
                outputs = generate_rollout_completions(trainer, [current_prompt])[0]
                episode_prompt_ids.extend(outputs["prompt_ids"])
                episode_completion_ids.extend(outputs["completion_ids"])
                episode_logprobs.extend(outputs["logprobs"])

                completion_text = outputs.get("text") or tokenizer.decode(
                    outputs["completion_ids"], skip_special_tokens=True
                )
                code = extract_cuda_code(completion_text)
                if not code:
                    reward = -1.0
                    result = {
                        "compiles": False,
                        "correct": False,
                        "error": (
                            "No valid CUDA/C++ code was found. Return a fenced code block "
                            "or a raw CUDA extension source file."
                        ),
                    }
                else:
                    compiles_locally, compile_err = _local_compile_check(code)
                    if not compiles_locally:
                        reward = -1.0
                        result = {"compiles": False, "correct": False, "error": compile_err[:200]}
                    elif not task_row.get("supports_evaluation"):
                        reward = -1.0
                        result = {
                            "compiles": False,
                            "correct": False,
                            "error": task_row.get("support_reason", "Unsupported evaluation backend"),
                        }
                    else:
                        try:
                            result = evaluate_code_remote(
                                code,
                                task_row,
                                baseline_orig_ms=baseline_orig,
                                baseline_dg_ms=baseline_dg,
                            )
                            reward = float(result.get("reward", _compute_reward_from_result(result)))
                        except Exception as exc:
                            print(f"  [Turn {turn + 1}] Eval dispatch failed: {exc}")
                            reward = -1.0
                            result = {"compiles": False, "correct": False, "error": str(exc)[:200]}

                if reward > best_reward:
                    best_reward = reward

                _append_rollout_log(
                    {
                        "prompt_index": prompt_idx,
                        "turn": turn + 1,
                        "evaluation_backend": task_row.get("evaluation_backend"),
                        "reward": reward,
                        "compiles": bool(result.get("compiles")),
                        "correct": bool(result.get("correct")),
                        "runtime_ms": float(result.get("runtime_ms", 0.0) or 0.0),
                        "speedup_vs_orig": float(result.get("speedup_vs_orig", 0.0) or 0.0),
                        "speedup_vs_dg": float(result.get("speedup_vs_dg", 0.0) or 0.0),
                    }
                )

                if reward >= 3.0 or turn == max_turns - 1:
                    break

                feedback = _format_feedback(result, reward, turn)
                current_prompt = build_generation_prompt(
                    task_row,
                    skill_context=skill_context,
                    topology_context=topology_ctx,
                ) + f"\n\n{feedback}"

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)
            all_best_rewards.append(best_reward)

            if (prompt_idx + 1) % 10 == 0 or prompt_idx == 0:
                print(
                    f"  Rollout {prompt_idx + 1}/{len(prompts)}: "
                    f"best_reward={best_reward:.3f} backend={task_row.get('evaluation_backend')}"
                )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_best_rewards,
        }

    return rollout_func


def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
    """Extract the rewards produced by rollout_func."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    return [-1.0] * len(completions)
