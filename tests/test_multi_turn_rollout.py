"""Tests for multi-turn rollout logic (no GPU/Modal needed)."""
from __future__ import annotations

import pytest

from training.multi_turn_rollout import (
    extract_cuda_code,
    _format_feedback,
    _compute_reward_from_result,
)


class TestExtractCudaCode:
    def test_fenced_cuda_block(self):
        text = "Here's the kernel:\n```cuda\n__global__ void add(float* a, float* b) {}\n```\nDone."
        assert "__global__ void add" in extract_cuda_code(text)

    def test_fenced_cpp_block(self):
        text = "```cpp\n__global__ void relu(float* x) {}\n```"
        assert "__global__ void relu" in extract_cuda_code(text)

    def test_raw_global(self):
        text = "__global__ void kernel(int* data, int n) { int i = threadIdx.x; }"
        assert "__global__ void kernel" in extract_cuda_code(text)

    def test_no_cuda_code(self):
        assert extract_cuda_code("This is just text, no CUDA here.") == ""

    def test_empty_string(self):
        assert extract_cuda_code("") == ""

    def test_multiple_blocks_takes_first(self):
        text = "```cuda\n__global__ void first() {}\n```\n```cuda\n__global__ void second() {}\n```"
        code = extract_cuda_code(text)
        assert "first" in code
        assert "second" not in code


class TestFormatFeedback:
    def test_compilation_failure(self):
        result = {"compiles": False, "error": "undefined reference to __shfl_sync"}
        feedback = _format_feedback(result, -1.0, 0)
        assert "COMPILATION FAILED" in feedback
        assert "__shfl_sync" in feedback
        assert "Fix" in feedback

    def test_verification_failure(self):
        result = {"compiles": True, "correct": False, "verifier_msg": "Invariant 2: edge (3,7) crosses components"}
        feedback = _format_feedback(result, -1.0, 1)
        assert "VERIFICATION FAILED" in feedback
        assert "edge (3,7)" in feedback

    def test_correct_slow(self):
        """speedup_vs_orig=0.8 → discrete reward 1.0 (correct but <1.05x) → 'not faster' tip."""
        reward = 1.0
        result = {"compiles": True, "correct": True, "runtime_ms": 5.0, "speedup_vs_orig": 0.8, "speedup_vs_dg": 0, "runtime_stats": {"mean": 5.1, "std": 0.2}}
        feedback = _format_feedback(result, reward, 2)
        assert "CORRECT" in feedback
        assert "5.000ms" in feedback
        assert "not faster" in feedback

    def test_correct_modest_speedup(self):
        """speedup_vs_orig=1.5, speedup_vs_dg=0.9 → discrete reward 2.0 → 'not torch.compile' tip."""
        reward = 2.0
        result = {"compiles": True, "correct": True, "runtime_ms": 2.0, "speedup_vs_orig": 1.5, "speedup_vs_dg": 0.9, "runtime_stats": {}}
        feedback = _format_feedback(result, reward, 3)
        assert "CORRECT" in feedback
        assert "Speedup vs eager: 1.50x" in feedback
        assert "torch.compile" in feedback


class TestComputeReward:
    def test_compile_fail(self):
        assert _compute_reward_from_result({"compiles": False}) == -1.0

    def test_verify_fail(self):
        assert _compute_reward_from_result({"compiles": True, "correct": False}) == -1.0

    def test_correct_slower(self):
        """speedup_vs_orig=0.9 → correct but <1.05x → discrete reward 1.0"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 0.9, "speedup_vs_dg": 0.5})
        assert r == 1.0

    def test_modest_speedup(self):
        """speedup_vs_orig=1.2 (>1.05x eager), speedup_vs_dg=0.9 (<1.05x compile) → reward 2.0"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 1.2, "speedup_vs_dg": 0.9})
        assert r == 2.0

    def test_large_speedup(self):
        """speedup_vs_orig=3.0, speedup_vs_dg=1.5 (>1.05x compile) → reward 3.0"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 3.0, "speedup_vs_dg": 1.5})
        assert r == 3.0
