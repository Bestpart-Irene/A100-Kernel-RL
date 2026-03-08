"""GPU architecture registry for KernelForge.

Maps target GPU names to hardware specifications used by the environment,
SKILL.md generation, and cross-compilation flags.

Source: DoubleGraph target_gpu.hpp pattern — compile-time GPU selection
via per-architecture specs. We make this runtime-configurable.
"""
from __future__ import annotations

from typing import Any

GPU_REGISTRY: dict[str, dict[str, Any]] = {
    "a100": {
        "name": "A100",
        "arch": "sm_80",
        "l2_cache_mb": 40,
        "smem_per_sm_kb": 164,
        "sms": 108,
        "cuda_cores": 6912,
        "hbm_bandwidth_tbs": 2.0,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_80",
        "has_tma": False,
        "has_dsmem": False,
        "has_dpx": False,
        "has_async_copy": True,
        "features": [
            "cp.async", "l2_persistence", "bf16_tensor_core",
            "tf32_tensor_core", "structured_sparsity_2_4",
        ],
    },
    "h200": {
        "name": "H200",
        "arch": "sm_90a",
        "l2_cache_mb": 50,
        "smem_per_sm_kb": 228,
        "sms": 132,
        "cuda_cores": 16896,
        "hbm_bandwidth_tbs": 4.8,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_90a",
        "has_tma": True,
        "has_dsmem": True,
        "has_dpx": True,
        "has_async_copy": True,
        "features": [
            "hbm3e", "tma", "dsmem", "dpx", "thread_block_clusters",
            "warp_specialization", "fp8", "cp.async", "l2_persistence",
            "bf16_tensor_core", "tf32_tensor_core",
        ],
    },
    "h100": {
        "name": "H100",
        "arch": "sm_90a",
        "l2_cache_mb": 50,
        "smem_per_sm_kb": 228,
        "sms": 132,
        "cuda_cores": 16896,
        "hbm_bandwidth_tbs": 3.35,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_90a",
        "has_tma": True,
        "has_dsmem": True,
        "has_dpx": True,
        "has_async_copy": True,
        "features": [
            "tma", "dsmem", "dpx", "thread_block_clusters",
            "warp_specialization", "fp8", "cp.async",
            "l2_persistence", "bf16_tensor_core", "tf32_tensor_core",
        ],
    },
    "b200": {
        "name": "B200",
        "arch": "sm_100a",
        "l2_cache_mb": 96,
        "smem_per_sm_kb": 228,
        "sms": 192,
        "cuda_cores": 18432,
        "hbm_bandwidth_tbs": 8.0,
        "registers_per_sm": 65536,
        "nvcc_flag": "-arch=sm_100a",
        "has_tma": True,
        "has_dsmem": True,
        "has_dpx": True,
        "has_async_copy": True,
        "features": [
            "tma", "dsmem", "dpx", "thread_block_clusters",
            "nvlink_5", "fp4", "fp8", "cp.async", "l2_persistence",
        ],
    },
}


def get_gpu_spec(gpu_name: str) -> dict[str, Any]:
    """Look up GPU spec by name. Raises ValueError if unknown."""
    key = gpu_name.lower()
    if key not in GPU_REGISTRY:
        raise ValueError(
            f"Unknown GPU: {gpu_name!r}. Available: {list(GPU_REGISTRY.keys())}"
        )
    return GPU_REGISTRY[key]
