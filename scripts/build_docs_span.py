#!/usr/bin/env python3
"""
Build a consolidated docs/codebase delta report ("SPAN") for KernelForge.

Usage:
  uv run python scripts/build_docs_span.py
  uv run python scripts/build_docs_span.py --output docs/KernelForge_Updated_SPAN.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

DOC_PATHS = {
    "PRD v5": ROOT / "docs" / "KernelForge_PRD_v5.md",
    "Implementation Spec": ROOT / "docs" / "KernelForge_Implementation_Spec.md",
    "Truth": ROOT / "docs" / "KernelForge_Truth.md",
}

EXPECTED_TRUTH_FILES = [
    "training/model_loader.py",
    "training/stage1_warmup.py",
    "training/stage2_rft.py",
    "training/stage3_grpo.py",
    "training/curriculum.py",
    "datasets/build_combined_dataset.py",
    "datasets/extract_doublegraph_a100.py",
    "datasets/integrity.py",
    "evaluation/eval_model.py",
    "evaluation/ablation.py",
    "tests/test_env.py",
    "tests/test_reward.py",
    "tests/test_compile.py",
]


@dataclass
class DocSignals:
    name: str
    target_gpu: str
    model: str
    training: str
    environment: str
    kernel_scope: str


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_line(lines: list[str], contains: str) -> int | None:
    for i, line in enumerate(lines, start=1):
        if contains in line:
            return i
    return None


def find_first_regex(lines: list[str], patterns: Iterable[str], default: str = "unknown") -> str:
    for pat in patterns:
        reg = re.compile(pat, flags=re.IGNORECASE)
        for line in lines:
            m = reg.search(line)
            if m:
                val = m.group(1).strip().strip("|").strip()
                return val
    return default


def find_table_value(lines: list[str], row_label: str, value_col: int = -1, default: str = "unknown") -> str:
    """Find a markdown table row by label and return the selected value column.

    Args:
        lines: document lines.
        row_label: first-column text to match (case-insensitive, markdown ** stripped).
        value_col: value column index among table value cells (0-based), or -1 for last.
    """
    needle = row_label.lower().strip()

    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue

        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cells) < 2:
            continue

        label = cells[0].replace("**", "").strip().lower()
        if needle != label:
            continue

        values = cells[1:]
        idx = value_col if value_col >= 0 else (len(values) - 1)
        if 0 <= idx < len(values):
            return values[idx].strip()

    return default


def summarize_doc(name: str, lines: list[str]) -> DocSignals:
    target_gpu = find_first_regex(
        lines,
        patterns=[
            r"\*\*Target GPU[^*]*:\*\*\s*(.+)$",
            r"\|\s*\*\*Target GPU \(kernels\)\*\*\s*\|\s*([^|]+)\|",
            r"\|\s*\*\*Hardware\*\*\s*\|\s*([^|]+)\|",
            r"\*\*Target Hardware\*\*\s*\|\s*([^|]+)\|",
        ],
    )
    model = find_first_regex(
        lines,
        patterns=[
            r"\*\*Agent Model:\*\*\s*(.+)$",
            r"\*\*Model:\*\*\s*(.+)$",
            r"\|\s*\*\*Model\*\*\s*\|\s*([^|]+)\|",
            r"\*\*Model\*\*\s*\|\s*([^|]+)\|",
        ],
    )
    training = find_first_regex(
        lines,
        patterns=[
            r"\|\s*\*\*Training approach\*\*\s*\|\s*([^|]+)\|",
            r"\*\*Training approach:\*\*\s*(.+)$",
            r"\*\*RL Framework:\*\*\s*(.+)$",
            r"Stage 1: (.+)$",
        ],
    )
    environment = find_first_regex(
        lines,
        patterns=[
            r"\*\*Environment:\*\*\s*(.+)$",
            r"\|\s*\*\*Environment\*\*\s*\|\s*([^|]+)\|",
        ],
    )
    kernel_scope = find_first_regex(
        lines,
        patterns=[
            r"\|\s*\*\*Kernel scope\*\*\s*\|\s*([^|]+)\|",
            r"\*\*Algorithm:\*\*\s*(.+)$",
        ],
    )

    # Document-specific extraction where the current-version values sit in table columns.
    if name == "PRD v4":
        target_gpu = find_table_value(lines, "Target GPU", value_col=1, default=target_gpu)
        model = find_table_value(lines, "Model", value_col=1, default=model)
        training = find_table_value(lines, "RL algorithm", value_col=1, default=training)
        environment = find_table_value(lines, "Environment", value_col=1, default=environment)
        kernel_scope = find_table_value(lines, "Algorithm scope", value_col=1, default=kernel_scope)

    if name == "PRD v5":
        if any("targeting A100 (sm_80)" in line for line in lines):
            target_gpu = "A100 (sm_80) kernels; H100/B200 training"
        model = find_table_value(lines, "Model", default=model)
        training = find_table_value(lines, "Training approach", default=training)
        environment = find_table_value(lines, "Environment", default=environment)
        kernel_scope = find_table_value(lines, "Kernel scope", default=kernel_scope)

    if name == "Implementation Spec":
        if "Cross-compiles for A100 (sm_80)" in "\n".join(lines):
            target_gpu = "Configurable target; A100 (sm_80) emphasized"
        if "Qwen3-Coder-Next" in "\n".join(lines):
            model = "Qwen3-Coder-Next primary (Qwen2.5 fallback path discussed)"
        if "adapted 3-stage pipeline" in "\n".join(lines) or "adapted 3-stage pipeline (no critic" in "\n".join(lines):
            training = "3-stage: warm-up -> RFT -> GRPO"
        if "OpenEnv API Conformance" in "\n".join(lines):
            environment = "OpenEnv local env server (Environment/StepResult API)"
        if "CUDA-Agent-Ops-6K" in "\n".join(lines):
            kernel_scope = "General CUDA operators (Ops-6K curated subset)"

    return DocSignals(
        name=name,
        target_gpu=target_gpu,
        model=model,
        training=training,
        environment=environment,
        kernel_scope=kernel_scope,
    )


def contains_text(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return needle in text


def line_ref(path: Path, needle: str) -> str:
    if not path.exists():
        return f"{path.relative_to(ROOT)}"
    line_no = find_line(read_lines(path), needle)
    if line_no is None:
        return f"{path.relative_to(ROOT)}"
    return f"{path.relative_to(ROOT)}#L{line_no}"


def codebase_snapshot() -> dict:
    env_file = ROOT / "openenv_env" / "kernel_forge_env.py"
    local_env_file = ROOT / "openenv_env" / "kernelforge_env.py"
    modal_file = ROOT / "modal_app.py"
    rft_file = ROOT / "training" / "rft_filter.py"
    grpo_file = ROOT / "training" / "grpo_train.py"
    sft_file = ROOT / "training" / "sft_warmup.py"
    cuda_agent_file = ROOT / "training" / "cuda_agent_integration.py"

    missing_truth_files = [
        rel for rel in EXPECTED_TRUTH_FILES if not (ROOT / rel).exists()
    ]

    return {
        "modal_backed_env": contains_text(env_file, "modal.Function.lookup"),
        "local_openenv_available": local_env_file.exists()
        and contains_text(local_env_file, "subprocess.run")
        and not contains_text(local_env_file, "modal.Function.lookup"),
        "target_a100_default": contains_text(env_file, "kernelforge-a100")
        and contains_text(modal_file, "sm_80"),
        "uses_ops6k_prompts": contains_text(cuda_agent_file, "load_dataset")
        and (
            contains_text(grpo_file, "load_cuda_agent_prompt_dataset")
            or contains_text(grpo_file, "load_cuda_agent_prompt_texts")
        )
        and contains_text(rft_file, "load_cuda_agent_prompt_texts"),
        "still_wcc_specific": contains_text(grpo_file, "WCC")
        or contains_text(sft_file, "WCC")
        or contains_text(rft_file, "WCC"),
        "has_local_truth_env": (ROOT / "openenv_env" / "kernelforge_env.py").exists(),
        "missing_truth_files": missing_truth_files,
        "refs": {
            "env_modal": line_ref(env_file, "modal.Function.lookup"),
            "local_env": line_ref(local_env_file, "subprocess.run"),
            "env_a100": line_ref(env_file, "kernelforge-a100"),
            "modal_arch": line_ref(modal_file, "TARGET_CUDA_ARCH"),
            "ops6k": line_ref(cuda_agent_file, "load_dataset"),
            "grpo_ops6k": (
                line_ref(grpo_file, "load_cuda_agent_prompt_dataset")
                if contains_text(grpo_file, "load_cuda_agent_prompt_dataset")
                else line_ref(grpo_file, "load_cuda_agent_prompt_texts")
            ),
            "rft_ops6k": line_ref(rft_file, "load_cuda_agent_prompt_texts"),
            "grpo_wcc": line_ref(grpo_file, "WCC"),
            "sft_wcc": line_ref(sft_file, "WCC"),
        },
    }


def build_markdown(signals: list[DocSignals], snapshot: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# KernelForge Updated SPAN (Spec + PRD + Codebase Alignment)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## 1) Decision evolution across docs")
    lines.append("")
    lines.append("| Document | Target GPU | Model | Training approach | Environment | Kernel scope |")
    lines.append("|---|---|---|---|---|---|")
    for s in signals:
        lines.append(
            f"| {s.name} | {s.target_gpu} | {s.model} | {s.training} | {s.environment} | {s.kernel_scope} |"
        )

    lines.append("")
    lines.append("## 2) Current codebase state (truth-alignment audit)")
    lines.append("")

    status_rows = [
        (
            "A100 default target",
            "✅ aligned" if snapshot["target_a100_default"] else "❌ missing",
            f"{snapshot['refs']['env_a100']}, {snapshot['refs']['modal_arch']}",
        ),
        (
            "Ops-6K prompt ingestion",
            "✅ aligned" if snapshot["uses_ops6k_prompts"] else "❌ missing",
            f"{snapshot['refs']['ops6k']}, {snapshot['refs']['grpo_ops6k']}, {snapshot['refs']['rft_ops6k']}",
        ),
        (
            "Local OpenEnv-only eval path (no Modal in step)",
            "✅ available" if snapshot["local_openenv_available"] else "❌ missing",
            snapshot["refs"]["local_env"],
        ),
        (
            "Legacy Modal-backed env remains in repo",
            "⚠️ present" if snapshot["modal_backed_env"] else "✅ absent",
            snapshot["refs"]["env_modal"],
        ),
        (
            "WCC-only coupling removed",
            "❌ still WCC-coupled" if snapshot["still_wcc_specific"] else "✅ mostly decoupled",
            f"{snapshot['refs']['grpo_wcc']}, {snapshot['refs']['sft_wcc']}",
        ),
        (
            "Truth repo layout files present",
            "❌ partial" if snapshot["missing_truth_files"] else "✅ complete",
            f"Missing: {len(snapshot['missing_truth_files'])}",
        ),
    ]

    lines.append("| Capability | Status | Evidence |")
    lines.append("|---|---|---|")
    for capability, status, evidence in status_rows:
        lines.append(f"| {capability} | {status} | `{evidence}` |")

    lines.append("")
    lines.append("## 3) Truth-vs-code gap list")
    lines.append("")

    if snapshot["missing_truth_files"]:
        lines.append("### Missing files expected by Truth Part 13")
        lines.append("")
        for rel in snapshot["missing_truth_files"]:
            lines.append(f"- [ ] `{rel}`")
        lines.append("")

    lines.append("### Behavioral mismatches")
    lines.append("")
    if not snapshot["local_openenv_available"]:
        lines.append("- [ ] Replace Modal-dependent env step path with local compile/verify/profile path for the truth workflow.")
    if snapshot["modal_backed_env"]:
        lines.append("- [ ] Decide and document default env path (legacy Modal env still present alongside local OpenEnv path).")
    if snapshot["still_wcc_specific"]:
        lines.append("- [ ] Decouple training prompts/data flow from WCC-only assumptions into multi-operator Ops-6K subsets.")
    lines.append("- [ ] Add model loader that supports Qwen3-Coder-Next primary + Qwen2.5 fallback path selection.")
    lines.append("- [ ] Add explicit curriculum module with difficulty phase promotion/demotion rules.")
    lines.append("- [ ] Add evaluation/ablation scripts for H1/H2/H3 claims in Truth Part 10.")

    lines.append("")
    lines.append("## 4) What to create and follow (execution order)")
    lines.append("")
    lines.append("### P0 — Must complete before hackathon run")
    lines.append("1. Implement missing training entrypoints (`model_loader.py`, `stage1_warmup.py`, `stage2_rft.py`, `stage3_grpo.py`).")
    lines.append("2. Build combined dataset (`build_combined_dataset.py` — merges doubleGraph + Ops-6K).")
    lines.append("3. Add local truth-style OpenEnv path (non-Modal) as an executable option.")
    lines.append("4. Add evaluation scripts (`evaluation/eval_model.py`, `evaluation/ablation.py`).")
    lines.append("5. Add smoke tests for reward/env/compile interfaces.")
    lines.append("")
    lines.append("### P1 — Strongly recommended")
    lines.append("1. Add anti-hack enforcement parity with truth (symbol scan + CU_FLAGS whitelist on local path).")
    lines.append("2. Add single-command runbook for Stage1 → Stage2 → Stage3.")
    lines.append("3. Persist stage metrics and ablation artifacts under `outputs/` for demo evidence.")
    lines.append("")
    lines.append("### P2 — Optional")
    lines.append("1. Add multi-architecture env registry parity from the Implementation Spec.")
    lines.append("2. Add richer dashboard wiring to consume real run outputs.")

    lines.append("")
    lines.append("## 5) Functional CLI usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# Rebuild this SPAN report")
    lines.append("uv run python scripts/build_docs_span.py")
    lines.append("")
    lines.append("# Write to a custom file")
    lines.append("uv run python scripts/build_docs_span.py --output docs/KernelForge_Updated_SPAN.md")
    lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build docs/codebase alignment SPAN markdown")
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs" / "KernelForge_Updated_SPAN.md"),
        help="Output markdown path",
    )
    args = parser.parse_args()

    signals: list[DocSignals] = []
    for name, path in DOC_PATHS.items():
        if not path.exists():
            continue
        lines = read_lines(path)
        signals.append(summarize_doc(name, lines))

    snapshot = codebase_snapshot()
    output_text = build_markdown(signals, snapshot)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")

    print(f"Wrote SPAN report to {output_path}")
    if snapshot["missing_truth_files"]:
        print(f"Missing truth-layout files: {len(snapshot['missing_truth_files'])}")


if __name__ == "__main__":
    main()
