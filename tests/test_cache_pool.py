"""Tests for GPU resource cache pool (LRU eviction)."""
import pytest

from openenv_env.cache_pool import GPUCachePool


def test_create_pool():
    pool = GPUCachePool(max_entries=4)
    assert len(pool) == 0


def test_invalid_max_entries():
    with pytest.raises(ValueError):
        GPUCachePool(max_entries=0)
    with pytest.raises(ValueError):
        GPUCachePool(max_entries=-1)


def test_get_or_create_stores_value():
    pool = GPUCachePool(max_entries=4)
    val = pool.get_or_create("key1", lambda: 42)
    assert val == 42
    assert len(pool) == 1


def test_get_or_create_returns_cached():
    pool = GPUCachePool(max_entries=4)
    pool.get_or_create("key1", lambda: 42)
    # Factory should NOT be called again
    val = pool.get_or_create("key1", lambda: 999)
    assert val == 42


def test_lru_eviction():
    pool = GPUCachePool(max_entries=2)
    pool.get_or_create("a", lambda: 1)
    pool.get_or_create("b", lambda: 2)
    pool.get_or_create("c", lambda: 3)  # evicts "a" (oldest)
    assert pool.get("a") is None
    assert pool.get("b") == 2
    assert pool.get("c") == 3
    assert len(pool) == 2


def test_lru_touch_reorders():
    pool = GPUCachePool(max_entries=2)
    pool.get_or_create("a", lambda: 1)
    pool.get_or_create("b", lambda: 2)
    pool.get("a")  # touch "a", making "b" the oldest
    pool.get_or_create("c", lambda: 3)  # evicts "b"
    assert pool.get("a") == 1
    assert pool.get("b") is None
    assert pool.get("c") == 3


def test_get_missing_returns_default():
    pool = GPUCachePool(max_entries=4)
    assert pool.get("missing") is None
    assert pool.get("missing", "fallback") == "fallback"


def test_clear():
    pool = GPUCachePool(max_entries=4)
    pool.get_or_create("a", lambda: 1)
    pool.get_or_create("b", lambda: 2)
    pool.clear()
    assert len(pool) == 0
    assert pool.get("a") is None


def test_eviction_calls_close():
    """Evicted values with a close() method should have it called."""
    closed = []

    class Closeable:
        def __init__(self, name):
            self.name = name
        def close(self):
            closed.append(self.name)

    pool = GPUCachePool(max_entries=1)
    pool.get_or_create("a", lambda: Closeable("a"))
    pool.get_or_create("b", lambda: Closeable("b"))  # evicts "a"
    assert "a" in closed
