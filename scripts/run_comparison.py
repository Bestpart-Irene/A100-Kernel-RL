#!/usr/bin/env python3
"""Multi-model scaling ladder for CUDA kernel generation.

Benchmarks N models against KernelBench (CUDA Agent Ops-6K), Dr. Kernel (WCC),
and combined suites. Measures whether small Opus-distilled models can match
CUDA Agent (230B MoE) quality for writing CUDA kernels.

Usage:
    # Baseline eval, all models, all benchmarks
    python scripts/run_comparison.py --models-file configs/scaling_ladder.json --eval-only

    # Specific models only
    python scripts/run_comparison.py --models-file configs/scaling_ladder.json \
        --models opus_2b,opus_9b --eval-only

    # Specific benchmark suite
    python scripts/run_comparison.py --models-file configs/scaling_ladder.json \
        --benchmark ops6k --eval-only

    # Full with RL training
    python scripts/run_comparison.py --models-file configs/scaling_ladder.json \
        --models opus_9b,coder_30b_moe --train-steps 50

    # Use Modal backend
    python scripts/run_comparison.py --models-file configs/scaling_ladder.json \
        --eval-backend modal --eval-only
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

# CUDA Agent reference numbers (arXiv 2602.24286, 230B MoE, 23B active)
CUDA_AGENT_REF = {
    "fast_0": 0.988,
    "fast_compile": 0.968,
    "geomean_vs_eager": 2.11,
}


def _detect_gpu_vram_gb() -> float:
    """Detect available GPU VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except ImportError:
        pass
    return 0.0


def _select_quant_bits(model_entry: dict, gpu_vram_gb: float, global_quant: str | int) -> int:
    """Select quantization for a model based on GPU VRAM."""
    if isinstance(global_quant, int):
        return global_quant
    # Auto mode
    bf16_gb = model_entry.get("bf16_gb", 100)
    overhead_gb = 20  # LoRA + optimizer + KV cache + activations
    if bf16_gb + overhead_gb < gpu_vram_gb:
        return 0  # bf16 fits
    return 8  # need 8-bit


def _load_models_config(config_path: str, model_filter: str | None = None) -> tuple[list[dict], dict]:
    """Load model registry and apply filters."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    models = config.get("models", config if isinstance(config, list) else [])
    settings = {k: v for k, v in config.items() if k != "models"} if isinstance(config, dict) else {}

    # Filter by enabled flag
    models = [m for m in models if m.get("enabled", True)]

    # Filter by --models flag
    if model_filter:
        labels = {l.strip() for l in model_filter.split(",")}
        models = [m for m in models if m["label"] in labels]
        missing = labels - {m["label"] for m in models}
        if missing:
            print(f"WARNING: models not found in config: {missing}")

    return models, settings


def _run_benchmark(
    model_name: str,
    output_path: str,
    quant_bits: int = 0,
    max_tasks: int | None = None,
    benchmark: str = "all",
    eval_backend: str | None = None,
) -> dict:
    """Run benchmark for a model via subprocess (clean GPU state)."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
        "--model", model_name,
        "--output", output_path,
        "--quant-bits", str(quant_bits),
        "--benchmark", benchmark,
    ]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])

    env = os.environ.copy()
    if eval_backend:
        env["KERNELFORGE_EVAL_BACKEND"] = eval_backend

    quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(quant_bits, f"{quant_bits}bit")
    print(f"\n  Benchmark: {model_name} ({quant_label}, {benchmark}) -> {output_path}")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return {}

    try:
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _run_training(
    model_name: str,
    quant_bits: int,
    output_prefix: str,
    train_steps: int = 50,
    eval_backend: str | None = None,
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
    if eval_backend:
        env["KERNELFORGE_EVAL_BACKEND"] = eval_backend

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


def _print_scaling_table(
    model_results: dict[str, dict],
    models: list[dict],
    phase: str = "baseline",
    benchmark: str = "all",
) -> None:
    """Print scaling summary table with CUDA Agent reference."""
    print(f"\n{'='*80}")
    print(f"SCALING LADDER — {phase.upper()} — benchmark={benchmark}")
    print(f"{'='*80}")
    print(f"  {'Model':<20} {'Params':>8} {'Type':>6} {'fast_0':>7} {'fast_comp':>10} "
          f"{'geomean':>8} {'% CUDA-Agent':>13}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*13}")

    # CUDA Agent reference line
    ref = CUDA_AGENT_REF
    print(f"  {'CUDA-Agent 230B':<20} {'23B-act':>8} {'MoE':>6} "
          f"{ref['fast_0']:>7.3f} {ref['fast_compile']:>10.3f} "
          f"{ref['geomean_vs_eager']:>7.2f}x {'100%':>13}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*13}")

    for m in models:
        label = m["label"]
        result = model_results.get(label, {})
        metrics = result.get("metrics", {})
        params = m.get("params_b", "?")
        mtype = m.get("type", "?")

        fast_0 = metrics.get("fast_0", 0)
        fast_c = metrics.get("fast_compile", 0)
        geomean = metrics.get("geomean_vs_eager", 0)
        pct = f"{fast_0 / ref['fast_0'] * 100:.0f}%" if fast_0 > 0 else "—"

        print(f"  {label:<20} {str(params)+'B':>8} {mtype:>6} "
              f"{fast_0:>7.3f} {fast_c:>10.3f} "
              f"{geomean:>7.2f}x {pct:>13}")

    print(f"{'='*80}")


def _print_rl_gains(
    baseline_results: dict[str, dict],
    post_rl_results: dict[str, dict],
    models: list[dict],
) -> None:
    """Print RL training gains per model."""
    print(f"\n{'='*80}")
    print("RL TRAINING GAINS (post-RL minus baseline)")
    print(f"{'='*80}")
    print(f"  {'Model':<20} {'fast_0':>14} {'fast_compile':>14} {'geomean':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14}")

    for m in models:
        label = m["label"]
        bm = baseline_results.get(label, {}).get("metrics", {})
        pm = post_rl_results.get(label, {}).get("metrics", {})
        if not bm or not pm:
            continue

        for metric in [("fast_0", "fast_0"), ("fast_compile", "fast_compile"), ("geomean_vs_eager", "geomean")]:
            pass  # handled inline below

        d_f0 = pm.get("fast_0", 0) - bm.get("fast_0", 0)
        d_fc = pm.get("fast_compile", 0) - bm.get("fast_compile", 0)
        d_gm = pm.get("geomean_vs_eager", 0) - bm.get("geomean_vs_eager", 0)

        def _fmt(d: float) -> str:
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.3f}"

        print(f"  {label:<20} {_fmt(d_f0):>14} {_fmt(d_fc):>14} {_fmt(d_gm):>14}")

    print(f"{'='*80}")


def run_comparison(
    models: list[dict],
    output_dir: str = "results/scaling_ladder",
    max_eval_tasks: int | None = None,
    train_steps: int = 50,
    eval_only: bool = False,
    benchmark: str = "all",
    eval_backend: str | None = None,
    global_quant: str | int = "auto",
):
    """Run multi-model scaling ladder comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Detect GPU and select quantization per model
    gpu_vram = _detect_gpu_vram_gb()
    gpu_label = f"{gpu_vram:.0f}GB" if gpu_vram > 0 else "no GPU detected"

    print("=" * 80)
    print("KERNELFORGE SCALING LADDER — Can small models match CUDA Agent?")
    print("=" * 80)
    print(f"  GPU: {gpu_label}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Eval backend: {eval_backend or 'default (coreweave)'}")
    print(f"  Models: {len(models)}")
    print()

    # Assign quantization per model
    for m in models:
        m["_quant_bits"] = _select_quant_bits(m, gpu_vram, global_quant)
        quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(m["_quant_bits"], "?")
        print(f"  {m['label']:<20} {str(m.get('params_b','?'))+'B':>8} {m.get('type','?'):>6} -> {quant_label}")
    print()

    benchmarks = ["ops6k", "wcc", "all"] if benchmark == "all" else [benchmark]

    # Phase 1: Baseline evaluation
    print("\n### Phase 1: Baseline Evaluation (no RL) ###")
    baseline_results: dict[str, dict[str, dict]] = {}  # {benchmark: {label: result}}

    for bm in benchmarks:
        baseline_results[bm] = {}
        for m in models:
            label = m["label"]
            out_path = f"{output_dir}/{label}_baseline_{bm}.json"
            result = _run_benchmark(
                m["model_id"], out_path,
                quant_bits=m["_quant_bits"],
                max_tasks=max_eval_tasks,
                benchmark=bm,
                eval_backend=eval_backend,
            )
            baseline_results[bm][label] = result

        _print_scaling_table(baseline_results[bm], models, phase="baseline", benchmark=bm)

    if eval_only:
        _save_summary(baseline_results, {}, models, output_dir, benchmarks)
        return

    # Phase 2: Train all models
    print("\n### Phase 2: Training (3-stage RL pipeline) ###")
    checkpoints: dict[str, str | None] = {}
    for m in models:
        label = m["label"]
        print(f"\n--- Training {label} ---")
        ckpt = _run_training(
            m["model_id"], m["_quant_bits"],
            output_prefix=f"{output_dir}/training/{label}",
            train_steps=train_steps,
            eval_backend=eval_backend,
        )
        checkpoints[label] = ckpt

    # Phase 3: Post-RL evaluation
    print("\n### Phase 3: Post-RL Evaluation ###")
    post_rl_results: dict[str, dict[str, dict]] = {}

    for bm in benchmarks:
        post_rl_results[bm] = {}
        for m in models:
            label = m["label"]
            ckpt = checkpoints.get(label)
            if not ckpt:
                continue
            out_path = f"{output_dir}/{label}_post_rl_{bm}.json"
            result = _run_benchmark(
                ckpt, out_path,
                quant_bits=m["_quant_bits"],
                max_tasks=max_eval_tasks,
                benchmark=bm,
                eval_backend=eval_backend,
            )
            post_rl_results[bm][label] = result

        _print_scaling_table(post_rl_results[bm], models, phase="post-rl", benchmark=bm)

    # Phase 4: RL gains
    print("\n### Phase 4: RL Training Gains ###")
    for bm in benchmarks:
        print(f"\n--- Benchmark: {bm} ---")
        _print_rl_gains(baseline_results.get(bm, {}), post_rl_results.get(bm, {}), models)

    _save_summary(baseline_results, post_rl_results, models, output_dir, benchmarks)


def _save_summary(
    baseline_results: dict[str, dict[str, dict]],
    post_rl_results: dict[str, dict[str, dict]],
    models: list[dict],
    output_dir: str,
    benchmarks: list[str],
) -> None:
    """Save complete scaling summary to JSON."""
    summary: dict = {
        "cuda_agent_reference": CUDA_AGENT_REF,
        "models": [],
        "benchmarks": {},
    }

    for m in models:
        summary["models"].append({
            "label": m["label"],
            "model_id": m["model_id"],
            "type": m.get("type"),
            "params_b": m.get("params_b"),
            "quant_bits": m.get("_quant_bits", 0),
        })

    for bm in benchmarks:
        bm_data: dict = {}
        for m in models:
            label = m["label"]
            entry: dict = {}
            bl = baseline_results.get(bm, {}).get(label, {})
            if bl:
                entry["baseline"] = bl.get("metrics", {})
            pr = post_rl_results.get(bm, {}).get(label, {})
            if pr:
                entry["post_rl"] = pr.get("metrics", {})
            if entry:
                bm_data[label] = entry
        summary["benchmarks"][bm] = bm_data

    summary_path = f"{output_dir}/scaling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nScaling summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model scaling ladder for CUDA kernel generation"
    )
    parser.add_argument("--models-file", required=True,
                        help="Path to model registry JSON (e.g., configs/scaling_ladder.json)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model labels to run (default: all enabled)")
    parser.add_argument("--quant-bits", type=str, default="auto",
                        help="Quantization: auto, 0 (bf16), 4 (NF4), 8 (INT8)")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "ops6k", "wcc"],
                        help="Benchmark suite: all (ops6k+wcc+combined), ops6k, wcc")
    parser.add_argument("--eval-backend", type=str, default=None, choices=["coreweave", "modal"],
                        help="Eval backend (overrides config)")
    parser.add_argument("--output-dir", default="results/scaling_ladder",
                        help="Directory for all outputs")
    parser.add_argument("--max-eval-tasks", type=int, default=None,
                        help="Limit eval to N tasks per benchmark (faster iteration)")
    parser.add_argument("--train-steps", type=int, default=50,
                        help="Training steps per stage (default: 50)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only compare baseline generation quality")
    args = parser.parse_args()

    # Parse quant_bits
    quant: str | int = args.quant_bits
    if quant != "auto":
        quant = int(quant)

    # Load model config
    models, settings = _load_models_config(args.models_file, model_filter=args.models)
    if not models:
        print("ERROR: No models selected. Check --models-file and --models flags.")
        sys.exit(1)

    # Apply config defaults, CLI overrides
    eval_backend = args.eval_backend or settings.get("eval_backend")

    run_comparison(
        models=models,
        output_dir=args.output_dir,
        max_eval_tasks=args.max_eval_tasks,
        train_steps=args.train_steps,
        eval_only=args.eval_only,
        benchmark=args.benchmark,
        eval_backend=eval_backend,
        global_quant=quant,
    )


if __name__ == "__main__":
    main()
