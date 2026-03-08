"""Tests for reward distribution monitoring (discrete rewards {-1, 1, 2, 3})."""
from evaluation.reward_monitor import check_reward_distribution


def test_empty_rewards():
    result = check_reward_distribution([])
    assert len(result["flags"]) > 0
    assert result["entropy"] == 0.0


def test_healthy_distribution():
    rewards = [-1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    result = check_reward_distribution(rewards)
    assert len(result["flags"]) == 0
    assert result["entropy"] > 0


def test_all_max_reward_flagged():
    rewards = [3.0] * 100
    result = check_reward_distribution(rewards)
    assert any("reward hacking" in f.lower() or "SUSPICIOUS" in f for f in result["flags"])


def test_bimodal_flagged():
    rewards = [-1.0] * 50 + [3.0] * 50
    result = check_reward_distribution(rewards)
    assert any("bimodal" in f.lower() for f in result["flags"])


def test_collapsed_flagged():
    rewards = [-1.0] * 100
    result = check_reward_distribution(rewards)
    assert any("collapsed" in f.lower() or "uniform" in f.lower() for f in result["flags"])


def test_no_positive_flagged():
    rewards = [-1.0] * 100
    result = check_reward_distribution(rewards)
    assert any("no positive" in f.lower() for f in result["flags"])


def test_tier_rates():
    rewards = [-1.0, 1.0, 2.0, 3.0]
    result = check_reward_distribution(rewards)
    assert result["tier_rates"]["fail_rate"] == 0.25
    assert result["tier_rates"]["correct_rate"] == 0.75
    assert result["tier_rates"]["speedup_eager_rate"] == 0.50
    assert result["tier_rates"]["speedup_compile_rate"] == 0.25
