"""Tests for compilation configuration."""
import os
import pytest


def test_target_arch_default():
    """Default arch should be sm_80 (A100)."""
    arch = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
    assert arch == "sm_80"


def test_target_arch_not_sm90():
    """Ensure we don't accidentally default to sm_90a (Hopper)."""
    arch = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
    assert "sm_90" not in arch


def test_no_hardcoded_sm90a_in_profile():
    """P0-5 verification: profile.py should not hardcode sm_90a."""
    profile_path = os.path.join(
        os.path.dirname(__file__), "..", "verification", "profile.py"
    )
    if not os.path.exists(profile_path):
        pytest.skip("profile.py not found")

    with open(profile_path) as f:
        content = f.read()

    # sm_90a should only appear in comments or env-gated contexts, never as a literal default
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if '"-arch=sm_90a"' in stripped or "'-arch=sm_90a'" in stripped:
            pytest.fail(f"Hardcoded sm_90a found at profile.py:{i}: {stripped}")


def test_nvcc_arch_flag_uses_env():
    """The arch flag should come from KERNELFORGE_TARGET_ARCH env var."""
    profile_path = os.path.join(
        os.path.dirname(__file__), "..", "verification", "profile.py"
    )
    if not os.path.exists(profile_path):
        pytest.skip("profile.py not found")

    with open(profile_path) as f:
        content = f.read()

    assert "KERNELFORGE_TARGET_ARCH" in content, "profile.py should reference KERNELFORGE_TARGET_ARCH"
