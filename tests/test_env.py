"""Tests for KernelForge OpenEnv environment contract."""
import pytest
from unittest.mock import patch, MagicMock

from openenv_env import KernelForgeEnv, KernelForgeAction, KernelForgeObservation


@pytest.fixture
def env():
    """Create env with mocked Modal backend."""
    with patch("openenv_env.kernel_forge_env.KernelForgeEnv._dispatch") as mock_modal:
        mock_modal.return_value = {"original_ms": 10.0, "doublegraph_ms": 8.0}
        e = KernelForgeEnv()
        yield e, mock_modal


def test_reset_returns_observation(env):
    e, mock_modal = env
    obs = e.reset()
    assert isinstance(obs, KernelForgeObservation)
    assert obs.turn == 0
    assert obs.done is False
    assert obs.text  # Should contain SKILL.md content


def test_reset_profiles_baselines(env):
    e, mock_modal = env
    # Force WCC task — baseline profiling only runs for WCC evaluation backend
    from openenv_env.task_pool import TaskPool
    e.task_pool = TaskPool([{
        "task_id": "test_wcc",
        "prompt": "WCC task",
        "ops": ["wcc"],
    }])
    e.reset()
    assert e.original_baseline_ms == 10.0
    assert e.doublegraph_baseline_ms == 8.0


def test_step_compile_failure(env):
    e, mock_modal = env
    e.reset()

    mock_modal.return_value = {"compiles": False, "error": "nvcc error"}
    action = KernelForgeAction(cuda_code="invalid code")
    obs = e.step(action)

    assert obs.reward == -1.0
    assert obs.turn == 1
    assert "COMPILATION FAILED" in obs.text


def test_step_verification_failure(env):
    e, mock_modal = env
    e.reset()

    mock_modal.return_value = {
        "compiles": True,
        "correct": False,
        "verifier_msg": "output mismatch",
    }
    action = KernelForgeAction(cuda_code="__global__ void k(){}")
    obs = e.step(action)

    assert obs.reward == -1.0
    assert "VERIFICATION FAILED" in obs.text


def test_step_correct_kernel(env):
    e, mock_modal = env
    e.reset()

    mock_modal.return_value = {
        "compiles": True,
        "correct": True,
        "runtime_ms": 12.0,
        "runtime_stats": {"median": 12.0},
    }
    action = KernelForgeAction(cuda_code="__global__ void k(){}")
    obs = e.step(action)

    # speedup = 10.0/12.0 ≈ 0.833, not > 1.05 → discrete reward 1.0
    assert obs.reward == 1.0
    assert "BENCHMARK" in obs.text


def test_step_fast_kernel(env):
    e, mock_modal = env
    e.reset()
    # Set eager baseline explicitly and clear doublegraph so only eager comparison applies
    e.original_baseline_ms = 10.0
    e.doublegraph_baseline_ms = None

    mock_modal.return_value = {
        "compiles": True,
        "correct": True,
        "runtime_ms": 5.0,  # 10.0/5.0 = 2.0x speedup vs eager
        "runtime_stats": {"median": 5.0},
    }
    action = KernelForgeAction(cuda_code="__global__ void k(){}")
    obs = e.step(action)

    # speedup = 10.0/5.0 = 2.0 > 1.05 → discrete reward 2.0
    assert obs.reward == 2.0


def test_done_at_max_turns(env):
    e, mock_modal = env
    e.reset()
    e.max_turns = 2

    mock_modal.return_value = {"compiles": False, "error": "err"}
    action = KernelForgeAction(cuda_code="bad")

    obs1 = e.step(action)
    assert obs1.done is False

    obs2 = e.step(action)
    assert obs2.done is True


def test_state_serializable(env):
    e, mock_modal = env
    e.reset()
    state = e.state
    assert state.episode_id
    assert state.step_count == 0


def test_history_accumulates(env):
    e, mock_modal = env
    e.reset()

    mock_modal.return_value = {"compiles": False, "error": "err"}
    action = KernelForgeAction(cuda_code="bad")

    e.step(action)
    e.step(action)

    assert len(e.history) == 2
    assert e.history[0]["turn"] == 1
    assert e.history[1]["turn"] == 2
