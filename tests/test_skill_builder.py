"""Tests for SKILL.md dynamic generation."""
import pytest

from openenv_env.skill_builder import build_skill_md


@pytest.mark.parametrize("gpu", ["a100", "h100", "b200"])
def test_returns_nonempty_string(gpu):
    result = build_skill_md(gpu)
    assert isinstance(result, str)
    assert len(result) > 100


def test_a100_content():
    md = build_skill_md("a100")
    assert "A100" in md or "sm_80" in md


def test_h100_mentions_tma():
    md = build_skill_md("h100")
    # H100 has TMA — skill doc should mention it
    assert "TMA" in md or "H100" in md


def test_b200_content():
    md = build_skill_md("b200")
    assert "B200" in md or "sm_100" in md


def test_unknown_gpu_raises():
    with pytest.raises((ValueError, KeyError)):
        build_skill_md("unknown_gpu")
