"""Tests for discrete milestone reward computation {-1, 1, 2, 3}."""
import pytest

from openenv_env.reward import compute_reward, trloo_post_process


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_fail():
    assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_no_speedup():
    """Correct but speedup=1.0 (not > 1.05) -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
    assert r == 1.0


def test_modest_speedup():
    """speedup_vs_eager=1.5 > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=0.9)
    assert r == 2.0


def test_large_speedup():
    """speedup_vs_eager=3.0 > 1.05 but speedup_vs_compile=1.0 not > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=3.0, speedup_vs_compile=1.0)
    assert r == 2.0


def test_slower_than_baseline():
    """Correct but speedup=0.5 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3)
    assert r == 1.0


def test_very_slow_clamped():
    """Correct but speedup=0.01 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.01, speedup_vs_compile=0)
    assert r == 1.0


def test_nsight_ignored():
    """Nsight metrics are accepted but unused in discrete mode — same reward as without."""
    base = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
    with_nsight = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=0.8, mem_coalescing=0.9, warp_efficiency=0.7,
    )
    assert with_nsight == base == 2.0


def test_nsight_extreme_values():
    """Nsight with out-of-range values still produces same discrete reward."""
    r = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=1.5, mem_coalescing=-0.1, warp_efficiency=0.5,
    )
    assert r == 2.0


def test_beats_torch_compile():
    """speedup_vs_compile=1.2 > 1.05 -> reward 3.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.2)
    assert r == 3.0


def test_trloo_post_process_g4():
    """TRLOO scales by N/(N-1) = 4/3 for G=4."""
    advantages = [0.5, -0.3, 1.2, -0.8]
    scaled = trloo_post_process(advantages, n=4)
    scale = 4 / 3
    for orig, result in zip(advantages, scaled):
        assert result == pytest.approx(orig * scale, abs=1e-6)


def test_trloo_post_process_g1():
    """TRLOO with N=1 returns unchanged."""
    advantages = [0.5]
    assert trloo_post_process(advantages, n=1) == advantages


def test_trloo_post_process_g2():
    """TRLOO scales by 2/1 = 2.0 for G=2."""
    advantages = [0.3, -0.3]
    scaled = trloo_post_process(advantages, n=2)
    assert scaled[0] == pytest.approx(0.6, abs=1e-6)
    assert scaled[1] == pytest.approx(-0.6, abs=1e-6)
