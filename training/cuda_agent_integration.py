"""Utilities to integrate CUDA-Agent-Ops-6K into KernelForge training prompts."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

CUDA_AGENT_DATASET_ID = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
TARGET_GPU_NAME = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATASETS_DIR = ROOT / "datasets"


def _load_hf_datasets():
    """Import the Hugging Face datasets package without shadowing the local datasets dir."""
    cwd = os.getcwd()
    orig_sys_path = list(sys.path)
    try:
        shadow_paths = {"", ".", cwd, str(ROOT), str(LOCAL_DATASETS_DIR)}
        sys.path = [path for path in sys.path if path not in shadow_paths]
        import datasets as hf_datasets  # noqa: WPS433

        return hf_datasets
    finally:
        sys.path = orig_sys_path


def _parse_ops(ops: Any) -> list[str]:
    """Best-effort parser for the `ops` field."""
    if isinstance(ops, list):
        return [str(x) for x in ops]

    if isinstance(ops, str):
        text = ops.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                continue

        return [text]

    return [str(ops)] if ops is not None else []


def _build_cuda_prompt(example: dict[str, Any], max_code_chars: int = 6000) -> str | None:
    """Convert one CUDA-Agent sample into a GRPO-compatible prompt."""
    code = str(example.get("code", "")).strip()
    if not code:
        return None

    ops = _parse_ops(example.get("ops"))
    ops_desc = ", ".join(ops) if ops else "unknown operator pipeline"
    data_source = str(example.get("data_source", "unknown"))

    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# [truncated]"

    return (
        f"Write a high-performance CUDA kernel implementation for NVIDIA {TARGET_GPU_NAME} ({TARGET_ARCH}) "
        "that matches the semantics of the PyTorch reference below.\n\n"
        f"Reference ops: {ops_desc}\n"
        f"Data source: {data_source}\n\n"
        "Reference implementation:\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        "Return a single CUDA/C++ source file only.\n"
        "The source must:\n"
        "- include `#include <torch/extension.h>`\n"
        "- define a callable `run_kernel(...)` entrypoint\n"
        "- export `run_kernel` via `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`\n"
        "- accept the tensors returned by `get_inputs()` and return outputs matching `Model(*inputs)`\n"
        "- include a `// CU_FLAGS:` comment if extra nvcc flags are required\n"
        "Do not return prose or Python wrappers."
    )


def load_cuda_agent_prompt_dataset(max_samples: int | None = 1024, seed: int = 42):
    """Load CUDA-Agent-Ops-6K and convert rows to prompt-only dataset."""
    hf_datasets = _load_hf_datasets()
    ds = hf_datasets.load_dataset(CUDA_AGENT_DATASET_ID, split="train")

    if max_samples is not None:
        max_samples = max(0, int(max_samples))
        if max_samples == 0:
            return hf_datasets.Dataset.from_list([])
        if len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

    prompts: list[dict[str, Any]] = []
    for row in ds:
        prompt = _build_cuda_prompt(row)
        if prompt:
            prompts.append({
                "prompt": prompt,
                "task_code": str(row.get("code", "")),
                "ops": str(row.get("ops", "")),
                "data_source": str(row.get("data_source", "")),
            })

    return hf_datasets.Dataset.from_list(prompts)


def load_cuda_agent_prompt_texts(max_samples: int = 128, seed: int = 42) -> list[str]:
    """Return CUDA-Agent prompts as a list of strings (for RFT prompt pools)."""
    ds = load_cuda_agent_prompt_dataset(max_samples=max_samples, seed=seed)
    return [str(row["prompt"]) for row in ds if row.get("prompt")]
