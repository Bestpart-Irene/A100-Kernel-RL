"""
Pure evaluation logic for CUDA kernel compilation, verification, and benchmarking.

Zero Modal dependency — used by both the Modal app (modal_app.py) and the
CoreWeave/Northflank FastAPI eval service (eval_service/app.py).
"""
import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np

from openenv_env.anti_hack import extract_cu_flags, scan_forbidden_symbols

# --- Configuration (from env vars, same defaults as modal_app.py) ---

TARGET_GPU = os.getenv("KERNELFORGE_MODAL_GPU", os.getenv("KERNELFORGE_TARGET_GPU", "A100"))
TARGET_CUDA_ARCH = os.getenv(
    "KERNELFORGE_CUDA_ARCH", os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
)
BASELINE_KERNEL = os.getenv("KERNELFORGE_BASELINE_KERNEL", "baseline_wcc.cu")
SECONDARY_BASELINE_KERNEL = os.getenv("KERNELFORGE_SECONDARY_BASELINE_KERNEL", "")
DEBUG_TIMINGS = os.getenv("KERNELFORGE_DEBUG_TIMINGS", "0") == "1"
_OPS_REFERENCE_CACHE_LIMIT = int(os.getenv("KERNELFORGE_OPS_REFERENCE_CACHE_LIMIT", "16"))
_OPS_REFERENCE_CACHE: dict[str, dict[str, Any]] = {}


# --- Helper functions ---


def _resolve_kernel_path(kernel_name: str) -> str | None:
    """Resolve kernel source path from common locations."""
    if not kernel_name:
        return None

    if os.path.isabs(kernel_name) and os.path.exists(kernel_name):
        return kernel_name

    root = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from eval_service/ to project root
    project_root = os.path.dirname(root)
    candidates = [
        os.path.join(project_root, "kernels", kernel_name),
        os.path.join(project_root, kernel_name),
        os.path.join("/root/kernels", kernel_name),
        os.path.join(os.getcwd(), "kernels", kernel_name),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _load_kernel_source(kernel_name: str) -> str:
    """Load CUDA kernel source by filename."""
    resolved = _resolve_kernel_path(kernel_name)
    if not resolved:
        raise FileNotFoundError(f"Kernel source not found: {kernel_name}")

    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


def _nvcc_command(
    src_path: str,
    output_path: str,
    cuda_code: str,
    shared: bool = True,
) -> list[str]:
    """Build a safe nvcc command with whitelisted user flags."""
    extra_flags = extract_cu_flags(cuda_code)
    cmd = ["nvcc", f"-arch={TARGET_CUDA_ARCH}", "-O3", src_path, "-o", output_path]
    if shared:
        cmd.extend(["--shared", "-Xcompiler", "-fPIC"])
    cmd.extend(extra_flags or ["--use_fast_math"])
    return cmd


def _ops_task_has_empty_init_inputs(task_code: str) -> bool:
    """Best-effort stateless-task detection for the local Ops harness."""
    import ast

    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "get_init_inputs":
            continue
        returns = [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Return)]
        if not returns:
            return True
        value = returns[-1]
        if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 0:
            return True
        if isinstance(value, ast.Constant) and value.value is None:
            return True
        return False

    return True


def _ops_task_supported(task_code: str) -> bool:
    """Return True when the task can be executed by the current extension harness."""
    deny_tokens = (
        "nn.Conv",
        "nn.Linear",
        "nn.BatchNorm",
        "nn.LSTM",
        "nn.GRU",
        "nn.Embedding",
        "nn.Parameter",
        "nn.MultiheadAttention",
        "register_buffer(",
        "register_parameter(",
    )
    return _ops_task_has_empty_init_inputs(task_code) and not any(
        token in task_code for token in deny_tokens
    )


def _move_to_cuda(value, torch):
    """Recursively move nested tensors to CUDA."""
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, list):
        return [_move_to_cuda(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cuda(item, torch) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_cuda(item, torch) for key, item in value.items()}
    return value


def _clone_value(value):
    """Recursively clone nested tensor structures."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _assert_close(candidate, reference, torch) -> None:
    """Recursively compare nested outputs."""
    if isinstance(reference, torch.Tensor):
        torch.testing.assert_close(candidate, reference, rtol=1e-3, atol=1e-3)
        return
    if isinstance(reference, tuple):
        if not isinstance(candidate, tuple) or len(candidate) != len(reference):
            raise AssertionError("Tuple output shape mismatch")
        for cand_item, ref_item in zip(candidate, reference):
            _assert_close(cand_item, ref_item, torch)
        return
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(candidate) != len(reference):
            raise AssertionError("List output shape mismatch")
        for cand_item, ref_item in zip(candidate, reference):
            _assert_close(cand_item, ref_item, torch)
        return
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or set(candidate) != set(reference):
            raise AssertionError("Dict output keys mismatch")
        for key in reference:
            _assert_close(candidate[key], reference[key], torch)
        return
    if candidate != reference:
        raise AssertionError(f"Scalar mismatch: {candidate!r} != {reference!r}")


def _module_name(prefix: str, payload: str) -> str:
    """Create a deterministic, collision-resistant extension module name."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall time in milliseconds."""
    return round((perf_counter() - start) * 1000.0, 3)


def _empty_phase_timings() -> dict[str, float]:
    """Create the canonical phase timing payload."""
    return {
        "compile_ms": 0.0,
        "correctness_ms": 0.0,
        "benchmark_eager_ms": 0.0,
        "benchmark_compile_ms": 0.0,
        "benchmark_kernel_ms": 0.0,
        "total_eval_ms": 0.0,
    }


def _finalize_eval_result(result: dict[str, Any], total_start: float) -> dict[str, Any]:
    """Attach total timing and emit one debug line if requested."""
    phase_timings = result.setdefault("phase_timings", _empty_phase_timings())
    for key, value in _empty_phase_timings().items():
        phase_timings.setdefault(key, value)
    phase_timings["total_eval_ms"] = _elapsed_ms(total_start)

    if DEBUG_TIMINGS:
        print(
            f"[eval] trace={result.get('trace_id', '')} task={result.get('task_id', '')} "
            f"backend={result.get('evaluation_backend', '')} compiles={result.get('compiles', False)} "
            f"correct={result.get('correct', False)} phases={phase_timings}"
        )
    return result


def _normalize_init_inputs(init_inputs: Any) -> list[Any]:
    """Normalize get_init_inputs() into a list."""
    if init_inputs is None:
        return []
    if isinstance(init_inputs, (list, tuple)):
        return list(init_inputs)
    return [init_inputs]


def _value_signature(value: Any) -> Any:
    """Build a stable structure signature for nested benchmark inputs."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch import handled by caller
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, list):
        return tuple(_value_signature(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_value_signature(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _value_signature(item)) for key, item in value.items()))
    return ("scalar", type(value).__name__, repr(value))


def _get_ops_reference_entry(task_code: str, torch) -> dict[str, Any]:
    """Cache reference module/model artifacts across warm worker invocations."""
    cache_key = hashlib.sha256(task_code.encode("utf-8")).hexdigest()
    cached = _OPS_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module = ModuleType(f"ops_ref_{cache_key[:12]}")
    module.__dict__["__file__"] = f"<ops_ref_{cache_key[:12]}>"
    exec(task_code, module.__dict__)

    init_inputs = _normalize_init_inputs(module.get_init_inputs())
    ref_model = module.Model(*init_inputs).eval().cuda()
    cached = {
        "module": module,
        "ref_model": ref_model,
        "eager_baselines": {},
        "compile_baselines": {},
        "compiled_models": {},
    }
    _OPS_REFERENCE_CACHE[cache_key] = cached

    while len(_OPS_REFERENCE_CACHE) > _OPS_REFERENCE_CACHE_LIMIT:
        oldest_key = next(iter(_OPS_REFERENCE_CACHE))
        if oldest_key == cache_key:
            break
        _OPS_REFERENCE_CACHE.pop(oldest_key, None)

    return cached


def edges_to_csr(edges, num_vertices):
    """Convert edge list to CSR format."""
    adj = [[] for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    row_ptr = np.zeros(num_vertices + 1, dtype=np.int32)
    col_idx = []

    for i, neighbors in enumerate(adj):
        row_ptr[i + 1] = row_ptr[i] + len(neighbors)
        col_idx.extend(sorted(neighbors))

    return row_ptr, np.array(col_idx, dtype=np.int32)


def _benchmark_kernel_source(
    cuda_code: str,
    warmup_iters: int = 40,
    benchmark_runs: int = 30,
    num_vertices: int = 10000,
) -> dict:
    """Compile, verify, and benchmark a kernel source string once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "baseline.cu")
        lib_path = os.path.join(tmpdir, "baseline.so")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(cuda_code)

        proc = subprocess.run(
            _nvcc_command(src_path, lib_path, cuda_code),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Baseline compile failed: {proc.stderr[:1200]}")

        from verification.pac_verify import (
            generate_test_graphs,
            run_kernel_verification,
            verify_wcc,
        )
        import cupy as cp

        graphs = generate_test_graphs(num_vertices=num_vertices)
        edges = graphs[0][1]
        n_verts = graphs[0][2]

        labels = run_kernel_verification(lib_path, edges, n_verts)
        passed, msg = verify_wcc(labels, edges, n_verts)
        if not passed:
            raise RuntimeError(f"Baseline verification failed: {msg}")

        for _ in range(warmup_iters):
            run_kernel_verification(lib_path, edges, n_verts)
        cp.cuda.Device(0).synchronize()

        times = []
        for _ in range(benchmark_runs):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            run_kernel_verification(lib_path, edges, n_verts)
            end.record()
            end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))

        times_np = np.array(times)
        return {
            "runtime_ms": float(np.median(times_np)),
            "runtime_stats": {
                "mean": float(np.mean(times_np)),
                "median": float(np.median(times_np)),
                "std": float(np.std(times_np)),
                "min": float(np.min(times_np)),
                "max": float(np.max(times_np)),
            },
        }


def _generate_binding_cpp(cuda_code: str) -> str:
    """Generate a minimal PyTorch C++ binding for a CUDA kernel."""
    global_fns = re.findall(
        r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', cuda_code
    )

    if not global_fns:
        return '''
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // No __global__ functions found — empty binding
}
'''

    fn_name = global_fns[0][0]
    return f'''
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>

// Forward declaration
extern "C" void {fn_name}(/* params */);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.doc() = "Auto-generated binding for {fn_name}";
}}
'''


# --- Main evaluation functions (implementations) ---


def evaluate_kernel_impl(payload: dict) -> dict:
    """
    Compile, verify (PAC-reasoning), and benchmark CUDA kernel on target GPU.

    Input:
        cuda_code: str          - CUDA source
        verify_graphs: int      - number of test graphs (default 5)
        warmup_iters: int       - warmup iterations (default 50)
        benchmark_runs: int     - timed runs (default 30)

    Returns:
        compiles: bool
        correct: bool
        verifier_msg: str
        runtime_ms: float       - median of benchmark_runs
        runtime_stats: dict     - {mean, median, std, min, max}
        speedup_vs_orig: float
        speedup_vs_dg: float
        error: str
    """
    total_start = perf_counter()
    cuda_code = payload.get("cuda_code", "")
    baseline_orig_ms = payload.get("baseline_original_ms")
    baseline_dg_ms = payload.get("baseline_doublegraph_ms")
    trace_id = str(payload.get("trace_id", ""))
    task_id = str(payload.get("task_id", ""))
    evaluation_backend = str(payload.get("evaluation_backend", "wcc"))
    skip_benchmark = bool(payload.get("skip_benchmark", False))
    result = {
        "compiles": False, "correct": False, "verifier_msg": "",
        "runtime_ms": 0.0, "runtime_stats": {},
        "speedup_vs_orig": 0.0, "speedup_vs_dg": 0.0, "error": "",
        "trace_id": trace_id, "task_id": task_id,
        "evaluation_backend": evaluation_backend,
        "phase_timings": _empty_phase_timings(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.cu")
        lib_path = os.path.join(tmpdir, "kernel.so")

        with open(src_path, "w") as f:
            f.write(cuda_code)

        # Step 1: Compile for target architecture
        compile_start = perf_counter()
        try:
            proc = subprocess.run(
                _nvcc_command(src_path, lib_path, cuda_code),
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                result["error"] = proc.stderr[:2000]
                result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)
                return _finalize_eval_result(result, total_start)
        except subprocess.TimeoutExpired:
            result["error"] = "Compilation timed out (30s limit)"
            result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)
            return _finalize_eval_result(result, total_start)
        result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)

        forbidden_reason = scan_forbidden_symbols(lib_path)
        if forbidden_reason:
            result["error"] = forbidden_reason
            return _finalize_eval_result(result, total_start)

        result["compiles"] = True

        # Step 2: PAC Verification (5 adversarial graphs)
        correctness_start = perf_counter()
        try:
            from verification.pac_verify import generate_test_graphs, verify_wcc, run_kernel_verification

            graphs = generate_test_graphs(num_vertices=10000)
            ctypes.CDLL(lib_path)

            all_passed = True
            for graph_type, edges, n_verts in graphs:
                kernel_labels = run_kernel_verification(lib_path, edges, n_verts)
                passed, msg = verify_wcc(kernel_labels, edges, n_verts)
                if not passed:
                    result["verifier_msg"] = f"FAILED on {graph_type}: {msg}"
                    all_passed = False
                    break

            result["correct"] = all_passed
            if all_passed:
                result["verifier_msg"] = f"All {len(graphs)} graphs verified"
        except Exception as e:
            result["correct"] = False
            result["verifier_msg"] = f"Verification exception: {str(e)[:500]}"
            result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
            return _finalize_eval_result(result, total_start)
        result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)

        # Step 3: Benchmark with cudaEvent timing
        if result["correct"] and skip_benchmark:
            result["verifier_msg"] = f"{result['verifier_msg']} (benchmark skipped)"
        elif result["correct"]:
            benchmark_start = perf_counter()
            try:
                import cupy as cp

                warmup = payload.get("warmup_iters", 50)
                runs = payload.get("benchmark_runs", 30)

                edges = graphs[0][1]
                n_verts = graphs[0][2]

                for _ in range(warmup):
                    run_kernel_verification(lib_path, edges, n_verts)
                cp.cuda.Device(0).synchronize()

                times = []
                for _ in range(runs):
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()
                    start.record()
                    run_kernel_verification(lib_path, edges, n_verts)
                    end.record()
                    end.synchronize()
                    times.append(cp.cuda.get_elapsed_time(start, end))

                times = np.array(times)
                result["runtime_ms"] = float(np.median(times))
                result["runtime_stats"] = {
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                }

                if baseline_orig_ms and result["runtime_ms"] > 0:
                    result["speedup_vs_orig"] = float(baseline_orig_ms) / result["runtime_ms"]
                if baseline_dg_ms and result["runtime_ms"] > 0:
                    result["speedup_vs_dg"] = float(baseline_dg_ms) / result["runtime_ms"]
            except Exception as e:
                result["error"] = f"Benchmark exception: {str(e)[:500]}"
            result["phase_timings"]["benchmark_kernel_ms"] = _elapsed_ms(benchmark_start)

        # Step 4: Lightweight profiling (occupancy + register info from ptxas)
        try:
            ptxas_cmd = [
                "nvcc", f"-arch={TARGET_CUDA_ARCH}", "-O3",
                "--ptxas-options=-v", "-c", src_path, "-o", "/dev/null",
            ]
            ptxas_cmd.extend(extract_cu_flags(cuda_code) or ["--use_fast_math"])
            ptxas_proc = subprocess.run(
                ptxas_cmd,
                capture_output=True, text=True, timeout=15,
            )
            ptxas_out = ptxas_proc.stderr

            reg_match = re.search(r'Used (\d+) registers', ptxas_out)
            smem_match = re.search(r'(\d+) bytes smem', ptxas_out)
            regs_per_thread = int(reg_match.group(1)) if reg_match else None
            shared_mem_bytes = int(smem_match.group(1)) if smem_match else 0

            if regs_per_thread is not None:
                max_threads_by_regs = min(65536 // regs_per_thread, 2048)
                max_warps_by_regs = max_threads_by_regs // 32
                theoretical_max_warps = 64
                occupancy = min(max_warps_by_regs / theoretical_max_warps, 1.0)
                result["occupancy"] = round(occupancy, 3)
                result["regs_per_thread"] = regs_per_thread
                result["shared_mem_bytes"] = shared_mem_bytes

            branch_match = re.search(r'(\d+) branch', ptxas_out)
            if branch_match:
                n_branches = int(branch_match.group(1))
                warp_efficiency = max(0.0, 1.0 - n_branches * 0.02)
                result["warp_efficiency"] = round(warp_efficiency, 3)
        except Exception:
            pass  # Profiling is best-effort

    return _finalize_eval_result(result, total_start)


def profile_baselines_impl() -> dict:
    """Profile baseline kernels on the target GPU."""
    details: dict[str, Any] = {
        "target_gpu": TARGET_GPU,
        "target_arch": TARGET_CUDA_ARCH,
        "baseline_kernel": BASELINE_KERNEL,
    }

    original_ms = None
    doublegraph_ms = None

    try:
        baseline_code = _load_kernel_source(BASELINE_KERNEL)
        baseline_result = _benchmark_kernel_source(baseline_code)
        original_ms = baseline_result["runtime_ms"]
        details["baseline_runtime_stats"] = baseline_result["runtime_stats"]
    except Exception as e:
        details["baseline_error"] = str(e)

    if SECONDARY_BASELINE_KERNEL:
        details["secondary_kernel"] = SECONDARY_BASELINE_KERNEL
        try:
            secondary_code = _load_kernel_source(SECONDARY_BASELINE_KERNEL)
            secondary_result = _benchmark_kernel_source(secondary_code)
            doublegraph_ms = secondary_result["runtime_ms"]
            details["secondary_runtime_stats"] = secondary_result["runtime_stats"]
        except Exception as e:
            details["secondary_error"] = str(e)

    if original_ms is None:
        try:
            import networkx as nx
            import time
            from verification.pac_verify import generate_test_graphs

            graphs = generate_test_graphs(num_vertices=50000)
            edges = graphs[0][1]
            n_verts = graphs[0][2]

            graph = nx.Graph()
            graph.add_nodes_from(range(n_verts))
            graph.add_edges_from(edges)

            start_t = time.time()
            _ = list(nx.connected_components(graph))
            original_ms = (time.time() - start_t) * 1000.0
            details["fallback_source"] = "networkx_cpu_measured"
        except Exception as e:
            details["fallback_error"] = str(e)

    return {
        "original_ms": original_ms,
        "doublegraph_ms": doublegraph_ms,
        "baseline_source": details.get("fallback_source", "compiled_kernel"),
        "details": details,
    }


def test_gpu_features_impl() -> dict:
    """Test target GPU feature availability."""
    try:
        import cupy as cp

        device = cp.cuda.Device(0)
        cc = device.compute_capability
        major = int(str(cc)[:1])
        compute_cap = f"{major}.{str(cc)[1:]}" if len(str(cc)) > 1 else str(cc)

        tma_available = major >= 9
        dsmem_available = major >= 9
        dpx_available = major >= 9
        async_copy_available = major >= 8

        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"unknown")
        device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

        return {
            "target_gpu": TARGET_GPU,
            "target_arch": TARGET_CUDA_ARCH,
            "tma_available": tma_available,
            "dsmem_available": dsmem_available,
            "dpx_available": dpx_available,
            "async_copy_available": async_copy_available,
            "compute_capability": compute_cap,
            "device_name": device_name,
        }

    except Exception as e:
        return {
            "error": str(e),
            "target_gpu": TARGET_GPU,
            "target_arch": TARGET_CUDA_ARCH,
            "tma_available": False,
            "dsmem_available": False,
            "dpx_available": False,
            "async_copy_available": False,
            "compute_capability": "unknown",
        }


def evaluate_kernels_batch_impl(payloads: list[dict]) -> list[dict]:
    """Evaluate multiple kernels in a single call."""
    if DEBUG_TIMINGS:
        print(f"[eval-batch] payload_count={len(payloads)}")
    results = []
    for payload in payloads:
        try:
            if payload.get("evaluation_backend") == "ops6k" or payload.get("task_code"):
                result = evaluate_ops6k_kernel_impl(payload)
            else:
                result = evaluate_kernel_impl(payload)
            results.append(result)
        except Exception as e:
            results.append({
                "compiles": False, "correct": False, "verifier_msg": "",
                "runtime_ms": 0.0, "runtime_stats": {},
                "speedup_vs_orig": 0.0, "speedup_vs_dg": 0.0,
                "error": f"Batch eval exception: {str(e)[:500]}",
            })
    return results


def evaluate_ops6k_kernel_impl(payload: dict) -> dict:
    """
    Evaluate a CUDA kernel against a PyTorch reference from CUDA-Agent-Ops-6K.

    Input:
        cuda_code: str     - LLM-generated CUDA kernel source
        task_code: str     - PyTorch reference (Model class + get_inputs + get_init_inputs)
        warmup_iters: int  - warmup iterations (default 10)
        benchmark_runs: int - timed runs (default 10)

    Returns:
        compiles: bool
        correct: bool
        runtime_ms: float
        baseline_eager_ms: float
        baseline_compile_ms: float
        speedup_vs_orig: float
        speedup_vs_dg: float
        error: str
    """
    import torch

    total_start = perf_counter()
    cuda_code = payload.get("cuda_code", "")
    task_code = payload.get("task_code", "")
    warmup_iters = int(payload.get("warmup_iters", 10))
    benchmark_runs = int(payload.get("benchmark_runs", 10))
    skip_benchmark = bool(payload.get("skip_benchmark", False))
    trace_id = str(payload.get("trace_id", ""))
    task_id = str(payload.get("task_id", ""))
    evaluation_backend = str(payload.get("evaluation_backend", "ops6k"))

    result = {
        "compiles": False,
        "correct": False,
        "runtime_ms": 0.0,
        "runtime_stats": {},
        "baseline_eager_ms": 0.0,
        "baseline_compile_ms": 0.0,
        "speedup_vs_orig": 0.0,
        "speedup_vs_dg": 0.0,
        "verifier_msg": "",
        "error": "",
        "trace_id": trace_id,
        "task_id": task_id,
        "evaluation_backend": evaluation_backend,
        "phase_timings": _empty_phase_timings(),
    }

    if not cuda_code or not task_code:
        result["error"] = "Missing cuda_code or task_code"
        return _finalize_eval_result(result, total_start)
    if not _ops_task_supported(task_code):
        result["error"] = (
            "Unsupported Ops task for live evaluation: only stateless tasks with empty "
            "get_init_inputs() are executable with the current extension harness."
        )
        return _finalize_eval_result(result, total_start)

    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "kernel.cu")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(cuda_code)

        compile_start = perf_counter()
        try:
            import torch.utils.cpp_extension as cpp_ext

            build_dir = os.path.join(tmpdir, "build")
            os.makedirs(build_dir, exist_ok=True)

            module_name = _module_name("ops_kernel", cuda_code)
            extra_cuda_cflags = ["-O3", f"-arch={TARGET_CUDA_ARCH}"]
            extra_cuda_cflags.extend(extract_cu_flags(cuda_code) or ["--use_fast_math"])
            extension = cpp_ext.load(
                name=module_name,
                sources=[kernel_path],
                build_directory=build_dir,
                verbose=False,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=list(dict.fromkeys(extra_cuda_cflags)),
            )
            if not hasattr(extension, "run_kernel"):
                result["error"] = (
                    "Compiled extension does not expose run_kernel. "
                    "Return a PYBIND11_MODULE with m.def(\"run_kernel\", &run_kernel)."
                )
                result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)
                return _finalize_eval_result(result, total_start)
            result["compiles"] = True
        except Exception as exc:
            result["error"] = f"Compilation failed: {str(exc)[:1500]}"
            result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)
            return _finalize_eval_result(result, total_start)
        result["phase_timings"]["compile_ms"] = _elapsed_ms(compile_start)

        correctness_start = perf_counter()
        try:
            torch.manual_seed(42)
            cache_entry = _get_ops_reference_entry(task_code, torch)
            ref_mod = cache_entry["module"]
            ref_model = cache_entry["ref_model"]
        except Exception as exc:
            result["error"] = f"Reference model load failed: {str(exc)[:500]}"
            result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
            return _finalize_eval_result(result, total_start)

        try:
            # Collect outputs for anti-hack checks (need 2+ distinct input sets)
            anti_hack_candidate_outputs = []
            anti_hack_ref_outputs = []
            anti_hack_inputs = []

            for seed in range(5):
                torch.manual_seed(42 + seed)
                test_inputs = ref_mod.get_inputs()
                if not isinstance(test_inputs, (list, tuple)):
                    test_inputs = [test_inputs]
                test_inputs = _move_to_cuda(test_inputs, torch)
                ref_inputs = _clone_value(test_inputs)
                candidate_inputs = _clone_value(test_inputs)

                with torch.no_grad():
                    ref_output = ref_model(*ref_inputs)
                    candidate_output = extension.run_kernel(*candidate_inputs)

                _assert_close(candidate_output, ref_output, torch)

                # Stash first 2 for anti-hack checks
                if seed < 2:
                    anti_hack_candidate_outputs.append(_clone_value(candidate_output))
                    anti_hack_ref_outputs.append(_clone_value(ref_output))
                    anti_hack_inputs.append(_clone_value(test_inputs))

            result["correct"] = True
            result["verifier_msg"] = "Outputs matched reference for 5 random seeds"
        except Exception as exc:
            result["error"] = f"Correctness check failed: {str(exc)[:800]}"
            result["verifier_msg"] = result["error"]
            result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
            return _finalize_eval_result(result, total_start)

        # Anti-hack checks (Dr. Kernel-inspired)
        try:
            from openenv_env.anti_hack import (
                check_not_passthrough,
                check_output_not_constant,
                check_shapes_match,
            )

            # Shape check
            if anti_hack_ref_outputs:
                passed, reason = check_shapes_match(
                    anti_hack_candidate_outputs[0], anti_hack_ref_outputs[0]
                )
                if not passed:
                    result["correct"] = False
                    result["error"] = f"Anti-hack: {reason}"
                    result["verifier_msg"] = result["error"]
                    result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
                    return _finalize_eval_result(result, total_start)

            # Constant output check
            if len(anti_hack_candidate_outputs) >= 2:
                passed, reason = check_output_not_constant(
                    anti_hack_candidate_outputs[0], anti_hack_candidate_outputs[1]
                )
                if not passed:
                    result["correct"] = False
                    result["error"] = f"Anti-hack: {reason}"
                    result["verifier_msg"] = result["error"]
                    result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
                    return _finalize_eval_result(result, total_start)

            # Passthrough check
            if anti_hack_inputs:
                passed, reason = check_not_passthrough(
                    anti_hack_candidate_outputs[0], anti_hack_inputs[0]
                )
                if not passed:
                    result["correct"] = False
                    result["error"] = f"Anti-hack: {reason}"
                    result["verifier_msg"] = result["error"]
                    result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)
                    return _finalize_eval_result(result, total_start)
        except Exception:
            pass  # Anti-hack is best-effort, don't fail the whole eval
        result["phase_timings"]["correctness_ms"] = _elapsed_ms(correctness_start)

        if skip_benchmark:
            result["verifier_msg"] = f"{result['verifier_msg']} (benchmark skipped)"
            return _finalize_eval_result(result, total_start)

        try:
            torch.manual_seed(42)
            bench_inputs = ref_mod.get_inputs()
            if not isinstance(bench_inputs, (list, tuple)):
                bench_inputs = [bench_inputs]
            bench_inputs = _move_to_cuda(bench_inputs, torch)
            bench_key = (
                warmup_iters,
                benchmark_runs,
                _value_signature(bench_inputs),
            )
            eager_cache = cache_entry["eager_baselines"]
            compile_cache = cache_entry["compile_baselines"]
            compiled_models = cache_entry["compiled_models"]

            eager_timing_start = perf_counter()
            eager_ms = eager_cache.get(bench_key)
            if eager_ms is None:
                for _ in range(warmup_iters):
                    with torch.no_grad():
                        ref_model(*_clone_value(bench_inputs))
                torch.cuda.synchronize()

                eager_times = []
                for _ in range(benchmark_runs):
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    with torch.no_grad():
                        ref_model(*_clone_value(bench_inputs))
                    end_evt.record()
                    end_evt.synchronize()
                    eager_times.append(start_evt.elapsed_time(end_evt))
                eager_ms = float(np.median(eager_times))
                eager_cache[bench_key] = eager_ms
            result["phase_timings"]["benchmark_eager_ms"] = _elapsed_ms(eager_timing_start)
            result["baseline_eager_ms"] = eager_ms

            compile_timing_start = perf_counter()
            compile_ms = compile_cache.get(bench_key, 0.0)
            try:
                if bench_key not in compile_cache:
                    compiled_model = compiled_models.get(bench_key)
                    if compiled_model is None:
                        compiled_model = torch.compile(ref_model)
                        compiled_models[bench_key] = compiled_model

                    for _ in range(warmup_iters):
                        with torch.no_grad():
                            compiled_model(*_clone_value(bench_inputs))
                    torch.cuda.synchronize()

                    compile_times = []
                    for _ in range(benchmark_runs):
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        start_evt.record()
                        with torch.no_grad():
                            compiled_model(*_clone_value(bench_inputs))
                        end_evt.record()
                        end_evt.synchronize()
                        compile_times.append(start_evt.elapsed_time(end_evt))

                    compile_ms = float(np.median(compile_times))
                    compile_cache[bench_key] = compile_ms
            except Exception:
                compile_ms = 0.0
            result["phase_timings"]["benchmark_compile_ms"] = _elapsed_ms(compile_timing_start)
            result["baseline_compile_ms"] = compile_ms

            kernel_timing_start = perf_counter()
            for _ in range(warmup_iters):
                extension.run_kernel(*_clone_value(bench_inputs))
            torch.cuda.synchronize()

            kernel_times = []
            for _ in range(benchmark_runs):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                extension.run_kernel(*_clone_value(bench_inputs))
                end_evt.record()
                end_evt.synchronize()
                kernel_times.append(start_evt.elapsed_time(end_evt))

            kernel_times_np = np.array(kernel_times)
            kernel_ms = float(np.median(kernel_times_np))
            result["runtime_ms"] = kernel_ms
            result["runtime_stats"] = {
                "mean": float(np.mean(kernel_times_np)),
                "median": float(np.median(kernel_times_np)),
                "std": float(np.std(kernel_times_np)),
                "min": float(np.min(kernel_times_np)),
                "max": float(np.max(kernel_times_np)),
                "eager_ms": eager_ms,
                "compile_ms": compile_ms,
            }

            if eager_ms > 0 and kernel_ms > 0:
                result["speedup_vs_orig"] = eager_ms / kernel_ms
            if compile_ms > 0 and kernel_ms > 0:
                result["speedup_vs_dg"] = compile_ms / kernel_ms

            # No-op check: if runtime is suspiciously fast, the kernel skips computation
            try:
                from openenv_env.anti_hack import check_not_noop

                passed, reason = check_not_noop(kernel_ms)
                if not passed:
                    result["correct"] = False
                    result["error"] = f"Anti-hack: {reason}"
                    result["verifier_msg"] = result["error"]
                    result["speedup_vs_orig"] = 0.0
                    result["speedup_vs_dg"] = 0.0
            except Exception:
                pass
            result["phase_timings"]["benchmark_kernel_ms"] = _elapsed_ms(kernel_timing_start)
        except Exception as exc:
            result["error"] = f"Profiling failed: {str(exc)[:500]}"
            result["phase_timings"]["benchmark_kernel_ms"] = max(
                result["phase_timings"]["benchmark_kernel_ms"],
                0.0,
            )

    return _finalize_eval_result(result, total_start)
