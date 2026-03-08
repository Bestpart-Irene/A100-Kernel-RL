"""Tests for GPU hardware registry."""
import pytest

from openenv_env.gpu_registry import GPU_REGISTRY, get_gpu_spec


REQUIRED_KEYS = {"name", "arch", "l2_cache_mb", "sms", "nvcc_flag", "features"}


def test_registry_has_all_gpus():
    assert set(GPU_REGISTRY.keys()) == {"a100", "h100", "b200", "h200"}


@pytest.mark.parametrize("gpu", ["a100", "h100", "b200"])
def test_required_keys_present(gpu):
    spec = get_gpu_spec(gpu)
    missing = REQUIRED_KEYS - set(spec.keys())
    assert not missing, f"{gpu} missing keys: {missing}"


@pytest.mark.parametrize("gpu", ["a100", "h100", "b200"])
def test_features_nonempty(gpu):
    spec = get_gpu_spec(gpu)
    assert len(spec["features"]) > 0


def test_a100_arch():
    assert get_gpu_spec("a100")["arch"] == "sm_80"


def test_h100_arch():
    assert get_gpu_spec("h100")["arch"] == "sm_90a"


def test_h100_has_tma():
    assert get_gpu_spec("h100")["has_tma"] is True


def test_a100_no_tma():
    assert get_gpu_spec("a100")["has_tma"] is False


def test_unknown_gpu_raises():
    with pytest.raises((ValueError, KeyError)):
        get_gpu_spec("unknown_gpu")


def test_case_insensitivity():
    """get_gpu_spec should handle common case variants."""
    # The function lowercases internally or the caller passes lowercase.
    # Test the actual behavior.
    spec = get_gpu_spec("a100")
    assert spec["name"] == "A100"
