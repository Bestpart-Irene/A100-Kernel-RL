"""Tests for task routing and evaluator payload helpers."""

from __future__ import annotations

import importlib


OPS_TASK_CODE = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_inputs():
    return [torch.randn(8)]


def get_init_inputs():
    return []
""".strip()


def _reload_task_support(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import training.task_support as task_support

    return importlib.reload(task_support)


def test_build_modal_payload_uses_debug_budget(monkeypatch):
    task_support = _reload_task_support(
        monkeypatch,
        KERNELFORGE_DEBUG_TIMINGS="1",
        KERNELFORGE_SKIP_BENCHMARK="1",
    )

    _, payload = task_support.build_modal_payload(
        "__global__ void k() {}",
        {
            "task_id": "ops_task_1",
            "prompt": "Write a CUDA kernel",
            "task_code": OPS_TASK_CODE,
        },
        trace_id="trace-ops",
    )

    assert payload["warmup_iters"] == 1
    assert payload["benchmark_runs"] == 3
    assert payload["skip_benchmark"] is True
    assert payload["trace_id"] == "trace-ops"
    assert payload["task_id"] == "ops_task_1"
    assert payload["evaluation_backend"] == "ops6k"


def test_build_modal_payload_preserves_explicit_skip_override(monkeypatch):
    task_support = _reload_task_support(
        monkeypatch,
        KERNELFORGE_DEBUG_TIMINGS="1",
        KERNELFORGE_SKIP_BENCHMARK="1",
    )

    _, payload = task_support.build_modal_payload(
        '__global__ void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {}',
        {
            "task_id": "wcc_task_1",
            "prompt": "Write a weakly connected components CUDA kernel",
            "ops": ["wcc"],
        },
        skip_benchmark=False,
        trace_id="trace-wcc",
    )

    assert payload["warmup_iters"] == 1
    assert payload["benchmark_runs"] == 3
    assert payload["skip_benchmark"] is False
    assert payload["trace_id"] == "trace-wcc"
    assert payload["task_id"] == "wcc_task_1"
    assert payload["evaluation_backend"] == "wcc"


def test_normalize_eval_result_adds_phase_timings(monkeypatch):
    task_support = _reload_task_support(monkeypatch)

    result = task_support.normalize_eval_result({"compiles": True, "correct": True})

    assert result["trace_id"] == ""
    assert result["task_id"] == ""
    assert result["phase_timings"]["compile_ms"] == 0.0
    assert result["phase_timings"]["total_eval_ms"] == 0.0


def test_evaluate_code_remote_batch_uses_batch_endpoint(monkeypatch):
    task_support = _reload_task_support(monkeypatch)
    recorded: dict[str, object] = {}

    def fake_dispatch(fn_name, payload):
        recorded["fn_name"] = fn_name
        recorded["payload"] = payload
        return [
            {"compiles": True, "correct": True, "trace_id": "trace-a", "task_id": "ops_task_1"},
            {"compiles": False, "correct": False, "trace_id": "trace-b", "task_id": "wcc_task_1"},
        ]

    monkeypatch.setattr("openenv_env.eval_backend.dispatch_eval", fake_dispatch)

    results = task_support.evaluate_code_remote_batch(
        ["__global__ void k() {}", '__global__ void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {}'],
        [
            {"task_id": "ops_task_1", "prompt": "Write a CUDA kernel", "task_code": OPS_TASK_CODE},
            {"task_id": "wcc_task_1", "prompt": "Write a weakly connected components CUDA kernel", "ops": ["wcc"]},
        ],
        skip_benchmark=True,
        trace_ids=["trace-a", "trace-b"],
    )

    assert recorded["fn_name"] == "evaluate_kernels_batch"
    payloads = recorded["payload"]
    assert isinstance(payloads, list)
    assert payloads[0]["trace_id"] == "trace-a"
    assert payloads[1]["trace_id"] == "trace-b"
    assert payloads[0]["skip_benchmark"] is True
    assert results[0]["reward"] == 0.2
    assert results[1]["reward"] == -0.4
