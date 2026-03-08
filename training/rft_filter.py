"""Rejection fine-tuning data collection for evaluator-backed tasks."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.multi_turn_rollout import extract_cuda_code
from training.run_metadata import utc_timestamp_rfc3339
from training.task_support import (
    build_generation_prompt,
    evaluate_code_remote,
    normalize_task_row,
)
from training.curriculum import format_topology_context
from openenv_env.skill_builder import build_skill_md

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
DEFAULT_MODAL_APP = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
DEFAULT_MODEL_PATH = os.getenv("KERNELFORGE_RFT_MODEL_PATH", "outputs/kernelforge-stage1")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("KERNELFORGE_RFT_MAX_NEW_TOKENS", "2048"))


class TrajectoryCollector:
    """Collect and filter evaluator-backed trajectories for Stage 2."""

    def __init__(self, modal_app_name: str | None = None, model_path: str | None = None):
        self.modal_app_name = modal_app_name or DEFAULT_MODAL_APP
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.trajectories: list[dict[str, Any]] = []
        self._generator = None
        self._task_pool: list[dict[str, Any]] | None = None

    def collect_trajectories(self, num_trajectories: int = 100) -> list[dict[str, Any]]:
        """Collect trajectories by sampling supported tasks."""
        print(f"Collecting {num_trajectories} trajectories...")
        task_pool = self._get_task_pool()

        for idx in range(num_trajectories):
            task = random.choice(task_pool)
            trajectory = self._run_single_trajectory(task, idx)

            if trajectory:
                self.trajectories.append(trajectory)
                print(
                    f"Trajectory {idx + 1}/{num_trajectories}: "
                    f"reward={trajectory['reward']:.3f} backend={trajectory['evaluation_backend']}"
                )

            if (idx + 1) % 10 == 0:
                print(f"Collected {len(self.trajectories)} trajectories so far...")

        return self.trajectories

    def _get_task_pool(self) -> list[dict[str, Any]]:
        """Load and cache supported training tasks for RFT data collection."""
        if self._task_pool is not None:
            return self._task_pool

        rows = load_training_dataset(stage="stage3", ops6k_max=int(os.getenv("CUDA_AGENT_RFT_PROMPTS", "64")))
        self._task_pool = [normalize_task_row(row) for row in rows]
        if not self._task_pool:
            raise RuntimeError("No evaluator-backed tasks are available for RFT collection.")
        return self._task_pool

    def _run_single_trajectory(self, task: dict[str, Any], trajectory_id: int) -> dict[str, Any] | None:
        """Run one prompt -> generation -> evaluation trajectory."""
        try:
            prompt = build_generation_prompt(
                task,
                skill_context=build_skill_md(TARGET_GPU.lower()),
                topology_context=format_topology_context(task),
            )
            model_output = self._get_model_response(prompt)
            code = extract_cuda_code(model_output)
            if not code:
                raise RuntimeError("Model response did not contain CUDA/C++ code")

            result = evaluate_code_remote(code, task)

            return {
                "id": trajectory_id,
                "trajectory_id": trajectory_id,
                "prompt": prompt,
                "model_output": model_output,
                "reward": float(result["reward"]),
                "compiles": bool(result.get("compiles")),
                "correct": bool(result.get("correct")),
                "speedup_vs_orig": float(result.get("speedup_vs_orig", 0.0) or 0.0),
                "speedup_vs_dg": float(result.get("speedup_vs_dg", 0.0) or 0.0),
                "error": result.get("error", ""),
                "evaluation_backend": task.get("evaluation_backend"),
                "task_metadata": task,
                "timestamp": utc_timestamp_rfc3339(),
            }
        except Exception as exc:
            print(f"Error in trajectory {trajectory_id}: {exc}")
            return None

    def _get_generator(self):
        """Lazily initialize the text-generation pipeline."""
        if self._generator is not None:
            return self._generator

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self._generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return self._generator

    def _fallback_kernel_template(self) -> str:
        """Deterministic fallback used when a checkpoint is unavailable."""
        return """```cuda
#include <cuda_runtime.h>

extern "C" __global__ void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    labels[tid] = tid;
}
```"""

    def _get_model_response(self, prompt: str) -> str:
        """Generate a model response with a safe fallback."""
        try:
            generator = self._get_generator()
            outputs = generator(
                prompt,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
            )
            text = outputs[0]["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return text.strip()
        except Exception as exc:
            print(f"Model generation failed ({self.model_path}): {exc}. Using fallback template.")
            return self._fallback_kernel_template()

    def filter_trajectories(self, min_reward: float = 1.0) -> list[dict[str, Any]]:
        """Filter trajectories based on reward threshold."""
        filtered = [trajectory for trajectory in self.trajectories if trajectory["reward"] >= min_reward]
        total = len(self.trajectories)
        if total > 0:
            print(f"Filtered trajectories: {len(filtered)}/{total} ({len(filtered) / total * 100:.1f}%)")
        else:
            print("No trajectories collected.")
        return filtered

    def save_rft_dataset(self, filtered_trajectories: list[dict[str, Any]], output_path: str):
        """Save filtered trajectories as a conversational SFT dataset."""
        rft_examples = []
        for trajectory in filtered_trajectories:
            text = (
                "<|user|>\n"
                f"{trajectory['prompt']}\n"
                "<|assistant|>\n"
                f"{trajectory['model_output']}"
            )
            example = {
                "messages": [
                    {"role": "user", "content": trajectory["prompt"]},
                    {"role": "assistant", "content": trajectory["model_output"]},
                ],
                "text": text,
                "reward": trajectory["reward"],
                "compiles": trajectory["compiles"],
                "correct": trajectory["correct"],
                "speedup_vs_orig": trajectory["speedup_vs_orig"],
                "speedup_vs_dg": trajectory["speedup_vs_dg"],
                "evaluation_backend": trajectory["evaluation_backend"],
                "trajectory_id": trajectory.get("trajectory_id", trajectory["id"]),
                "timestamp": trajectory["timestamp"],
            }
            rft_examples.append(example)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in rft_examples:
                f.write(json.dumps(example) + "\n")

        print(f"Saved {len(rft_examples)} RFT examples to {output_path}")

        if hasattr(Dataset, "from_list"):
            dataset = Dataset.from_list(rft_examples)
            dataset.save_to_disk(output_path.replace(".jsonl", "_hf"))
        else:
            dataset = MiniDataset(rft_examples)
        return dataset


def main():
    """Main RFT filtering process."""
    print("Starting Rejection Fine-Tuning (RFT) filtering...")

    collector = TrajectoryCollector()
    trajectories = collector.collect_trajectories(num_trajectories=50)
    filtered = collector.filter_trajectories(min_reward=1.0)

    if not filtered:
        print("No trajectories met the quality threshold!")
        return

    os.makedirs("datasets", exist_ok=True)
    collector.save_rft_dataset(filtered, "datasets/rft_filtered.jsonl")

    rewards = [trajectory["reward"] for trajectory in filtered]
    print(
        "RFT filtering completed! "
        f"Collected {len(trajectories)} trajectories and kept {len(filtered)}.\n"
        f"Reward statistics: min={min(rewards):.3f}, max={max(rewards):.3f}, "
        f"mean={sum(rewards) / len(rewards):.3f}"
    )


if __name__ == "__main__":
    main()
