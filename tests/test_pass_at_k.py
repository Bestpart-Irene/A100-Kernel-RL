"""Tests for Pass@k metric (unbiased estimator)."""
import pytest

from evaluation.pass_at_k import pass_at_k, pass_at_k_problems


def test_all_correct():
    """If all samples pass, pass@k = 1.0 for any k."""
    assert pass_at_k(n=10, c=10, k=1) == 1.0
    assert pass_at_k(n=10, c=10, k=5) == 1.0


def test_none_correct():
    """If no samples pass, pass@k = 0.0."""
    assert pass_at_k(n=10, c=0, k=1) == 0.0
    assert pass_at_k(n=10, c=0, k=5) == 0.0


def test_one_of_ten():
    """pass@1 with 1/10 correct = 0.1."""
    result = pass_at_k(n=10, c=1, k=1)
    assert abs(result - 0.1) < 1e-9


def test_pass_at_k_increases_with_k():
    """pass@k should increase as k increases."""
    p1 = pass_at_k(n=20, c=5, k=1)
    p5 = pass_at_k(n=20, c=5, k=5)
    p10 = pass_at_k(n=20, c=5, k=10)
    assert p1 < p5 < p10


def test_invalid_n_less_than_k():
    with pytest.raises(ValueError):
        pass_at_k(n=3, c=1, k=5)


def test_invalid_c_greater_than_n():
    with pytest.raises(ValueError):
        pass_at_k(n=5, c=6, k=1)


def test_pass_at_k_problems():
    results = [
        {"n": 10, "c": 5},
        {"n": 10, "c": 0},
        {"n": 10, "c": 10},
    ]
    output = pass_at_k_problems(results, k_values=[1, 5])
    assert "pass@1" in output
    assert "pass@5" in output
    assert 0 <= output["pass@1"] <= 1.0
    assert output["pass@1_count"] == 3
