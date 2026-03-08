#!/usr/bin/env python3
"""Modal deployment health check and smoke test.

Usage:
    uv run python scripts/modal_setup.py              # Deploy + full smoke test
    uv run python scripts/modal_setup.py --check-only  # Just check token/install
    uv run python scripts/modal_setup.py --deploy-only  # Deploy without testing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")


def check_modal_installed() -> bool:
    """Check if modal SDK is installed."""
    try:
        import modal  # noqa: F401
        print(f"  modal SDK: installed (version {modal.__version__})")
        return True
    except ImportError:
        print("  modal SDK: NOT INSTALLED")
        print("  Fix: pip install 'modal>=0.70' or uv pip install 'modal>=0.70'")
        return False


def check_modal_token() -> bool:
    """Check if Modal token is configured."""
    result = subprocess.run(
        ["modal", "token", "info"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        print(f"  Modal token: active")
        return True
    else:
        print("  Modal token: NOT CONFIGURED")
        print("  Fix: modal token new")
        return False


def deploy_app() -> bool:
    """Deploy modal_app.py to Modal."""
    print(f"\nDeploying {APP_NAME} to Modal...")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(root, "modal_app.py")

    result = subprocess.run(
        ["modal", "deploy", app_path],
        cwd=root,
        timeout=600,
    )
    if result.returncode == 0:
        print(f"  Deploy: SUCCESS")
        return True
    else:
        print(f"  Deploy: FAILED (exit code {result.returncode})")
        return False


def test_gpu_features() -> dict | None:
    """Test GPU feature detection on remote hardware."""
    import modal

    print(f"\nTesting GPU features on {TARGET_GPU}...")
    try:
        fn = modal.Function.from_name(APP_NAME, "test_h100_features")
        result = fn.remote()
        print(f"  Device: {result.get('device_name', 'unknown')}")
        print(f"  Compute capability: {result.get('compute_capability', 'unknown')}")
        print(f"  Target arch: {result.get('target_arch', TARGET_ARCH)}")
        print(f"  TMA: {result.get('tma_available', False)}")
        print(f"  DSMEM: {result.get('dsmem_available', False)}")
        print(f"  DPX: {result.get('dpx_available', False)}")
        print(f"  Async copy: {result.get('async_copy_available', False)}")
        return result
    except Exception as e:
        print(f"  GPU test FAILED: {e}")
        return None


def test_baselines() -> dict | None:
    """Profile baseline kernels on remote hardware."""
    import modal

    print(f"\nProfiling baselines on {TARGET_GPU}...")
    try:
        fn = modal.Function.from_name(APP_NAME, "profile_baselines")
        result = fn.remote()
        orig = result.get("original_ms")
        dg = result.get("doublegraph_ms")
        src = result.get("baseline_source", "unknown")
        print(f"  Baseline source: {src}")
        print(f"  Original (eager): {orig:.3f} ms" if orig else "  Original: N/A")
        print(f"  DoubleGraph: {dg:.3f} ms" if dg else "  DoubleGraph: N/A")
        return result
    except Exception as e:
        print(f"  Baseline profiling FAILED: {e}")
        return None


def test_eval_kernel() -> dict | None:
    """Submit a trivial kernel to evaluate_kernel for end-to-end test."""
    import modal

    # Minimal vector-add kernel (should compile but won't pass WCC verification)
    trivial_kernel = """
#include <cuda_runtime.h>

extern "C" __global__ void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        labels[tid] = tid;  // Trivial: each vertex is its own component
    }
}
"""
    print(f"\nTesting evaluate_kernel (trivial kernel)...")
    try:
        fn = modal.Function.from_name(APP_NAME, "evaluate_kernel")
        result = fn.remote({
            "cuda_code": trivial_kernel,
            "verify_graphs": 5,
            "warmup_iters": 10,
            "benchmark_runs": 5,
        })
        print(f"  Compiles: {result.get('compiles', False)}")
        print(f"  Correct: {result.get('correct', False)}")
        print(f"  Verifier: {result.get('verifier_msg', 'N/A')[:80]}")
        if result.get("error"):
            print(f"  Error: {result['error'][:100]}")
        return result
    except Exception as e:
        print(f"  Eval test FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Modal deployment health check")
    parser.add_argument("--check-only", action="store_true", help="Only check install/token")
    parser.add_argument("--deploy-only", action="store_true", help="Deploy without testing")
    args = parser.parse_args()

    print(f"KernelForge Modal Setup — {TARGET_GPU} ({TARGET_ARCH})")
    print(f"App name: {APP_NAME}")
    print("=" * 50)

    # Step 1: Check prerequisites
    print("\n[1/4] Checking prerequisites...")
    if not check_modal_installed():
        sys.exit(1)
    if not check_modal_token():
        sys.exit(1)

    if args.check_only:
        print("\nPrerequisites OK.")
        return

    # Step 2: Deploy
    print("\n[2/4] Deploying app...")
    if not deploy_app():
        sys.exit(1)

    if args.deploy_only:
        print("\nDeployment complete.")
        return

    # Step 3: GPU smoke test
    print("\n[3/4] Running GPU smoke tests...")
    gpu_result = test_gpu_features()
    baseline_result = test_baselines()

    # Step 4: Evaluation pipeline test
    print("\n[4/4] Testing evaluation pipeline...")
    eval_result = test_eval_kernel()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = 0
    total = 3

    if gpu_result and not gpu_result.get("error"):
        print(f"  GPU features:  PASS ({gpu_result.get('device_name', '?')})")
        passed += 1
    else:
        print("  GPU features:  FAIL")

    if baseline_result and baseline_result.get("original_ms"):
        print(f"  Baselines:     PASS ({baseline_result['original_ms']:.3f} ms)")
        passed += 1
    else:
        print("  Baselines:     FAIL")

    if eval_result and eval_result.get("compiles"):
        print(f"  Eval pipeline: PASS (compiles={eval_result['compiles']})")
        passed += 1
    else:
        print("  Eval pipeline: FAIL")

    print(f"\n  {passed}/{total} checks passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
