"""Tests for anti-reward-hacking utilities."""
import pytest

from openenv_env.anti_hack import extract_cu_flags, FORBIDDEN_SYMBOLS, scan_forbidden_symbols


def test_empty_code():
    assert extract_cu_flags("") == []


def test_no_cu_flags_line():
    assert extract_cu_flags("__global__ void kernel() {}") == []


def test_single_allowed_flag():
    assert extract_cu_flags("// CU_FLAGS: --use_fast_math") == ["--use_fast_math"]


def test_multiple_allowed_flags():
    result = extract_cu_flags("// CU_FLAGS: --use_fast_math --extra-device-vectorization")
    assert "--use_fast_math" in result
    assert "--extra-device-vectorization" in result


def test_rdc_flag():
    assert extract_cu_flags("// CU_FLAGS: --rdc=true") == ["--rdc=true"]


def test_maxrregcount_valid():
    assert extract_cu_flags("// CU_FLAGS: --maxrregcount=48") == ["--maxrregcount=48"]


def test_maxrregcount_out_of_range_high():
    assert extract_cu_flags("// CU_FLAGS: --maxrregcount=256") == []


def test_maxrregcount_out_of_range_low():
    assert extract_cu_flags("// CU_FLAGS: --maxrregcount=8") == []


def test_disallowed_flag_rejected():
    assert extract_cu_flags("// CU_FLAGS: --dangerous-flag") == []


def test_duplicate_flags_deduplicated():
    code = "// CU_FLAGS: --use_fast_math\n// CU_FLAGS: --use_fast_math"
    result = extract_cu_flags(code)
    assert result == ["--use_fast_math"]


def test_forbidden_symbols_list():
    assert "torch" in FORBIDDEN_SYMBOLS
    assert "triton" in FORBIDDEN_SYMBOLS
    assert "at::Tensor" in FORBIDDEN_SYMBOLS


def test_scan_nonexistent_path():
    result = scan_forbidden_symbols("/nonexistent/path.so")
    # Should not raise; returns None or a failure string
    assert result is None or isinstance(result, str)
