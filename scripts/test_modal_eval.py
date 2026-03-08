#!/usr/bin/env python3
"""End-to-end Modal evaluation test with a real WCC kernel.

Submits the baseline WCC kernel from kernels/baseline_wcc.cu to Modal,
verifies it compiles, passes PAC verification, and benchmarks correctly.

Usage:
    uv run python scripts/test_modal_eval.py
"""
from __future__ import annotations

import os
import sys


APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
SKIP_BENCHMARK = os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1"


def load_baseline_kernel() -> str:
    """Load baseline WCC kernel source."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_path = os.path.join(root, "kernels", "baseline_wcc.cu")
    if not os.path.exists(kernel_path):
        print(f"Baseline kernel not found: {kernel_path}")
        sys.exit(1)
    with open(kernel_path) as f:
        return f.read()


def main():
    import modal

    print(f"End-to-end Modal evaluation test ({APP_NAME})")
    print("=" * 50)

    # Load the baseline kernel
    cuda_code = load_baseline_kernel()
    print(f"Loaded baseline kernel ({len(cuda_code)} chars)")

    # Get baselines first
    print("\nStep 1: Profile baselines...")
    baseline_fn = modal.Function.from_name(APP_NAME, "profile_baselines")
    baselines = baseline_fn.remote()
    orig_ms = baselines.get("original_ms")
    dg_ms = baselines.get("doublegraph_ms")
    print(f"  Original baseline: {orig_ms:.3f} ms" if orig_ms else "  Original: N/A (using CPU fallback)")
    print(f"  DoubleGraph baseline: {dg_ms:.3f} ms" if dg_ms else "  DoubleGraph: N/A")

    # Submit kernel for evaluation
    print("\nStep 2: Evaluate baseline kernel...")
    eval_fn = modal.Function.from_name(APP_NAME, "evaluate_kernel")
    result = eval_fn.remote({
        "cuda_code": cuda_code,
        "verify_graphs": 5,
        "warmup_iters": 50,
        "benchmark_runs": 30,
        "baseline_original_ms": orig_ms,
        "baseline_doublegraph_ms": dg_ms,
        "evaluation_backend": "wcc",
        "task_id": "test_modal_eval",
        "trace_id": "test_modal_eval",
        "skip_benchmark": SKIP_BENCHMARK,
    })

    # Validate results
    print("\nStep 3: Validate results...")
    checks = []

    compiles = result.get("compiles", False)
    correct = result.get("correct", False)
    runtime = result.get("runtime_ms", 0)
    verifier = result.get("verifier_msg", "")
    error = result.get("error", "")
    speedup_orig = result.get("speedup_vs_orig", 0)
    stats = result.get("runtime_stats", {})

    # Check 1: Compilation
    checks.append(("Compilation", compiles))
    print(f"  Compiles: {compiles}")
    if error:
        print(f"  Error: {error[:200]}")

    # Check 2: PAC Verification
    checks.append(("PAC Verification", correct))
    print(f"  Correct: {correct}")
    print(f"  Verifier: {verifier[:100]}")

    # Check 3: Runtime path matches mode
    runtime_ok = runtime > 0 if not SKIP_BENCHMARK else runtime == 0
    checks.append(("Runtime path", runtime_ok))
    print(f"  Runtime: {runtime:.3f} ms")

    # Check 4: Runtime stats present when benchmarking is enabled
    has_stats = bool(stats and stats.get("median", 0) > 0) if not SKIP_BENCHMARK else True
    checks.append(("Runtime stats", has_stats))
    if stats:
        print(f"  Stats: mean={stats.get('mean', 0):.3f}, median={stats.get('median', 0):.3f}, "
              f"std={stats.get('std', 0):.3f}")
    elif SKIP_BENCHMARK:
        print("  Stats: skipped (KERNELFORGE_SKIP_BENCHMARK=1)")

    # Check 5: Speedup is reasonable (not NaN or negative)
    speedup_ok = speedup_orig > 0 if orig_ms and not SKIP_BENCHMARK else True
    checks.append(("Speedup sanity", speedup_ok))
    print(f"  Speedup vs original: {speedup_orig:.2f}x")

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n  {passed}/{total} checks passed")

    if passed < total:
        print("\nSome checks failed. Review the output above.")
        sys.exit(1)
    else:
        print("\nAll checks passed! Modal evaluation pipeline is working.")


if __name__ == "__main__":
    main()
