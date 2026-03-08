"""GPU resource cache helpers inspired by DoubleGraph CachePool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GPUCacheEntry:
    """Container for cached value plus optional metadata."""

    key: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class GPUCachePool:
    """Simple LRU cache for GPU-side resources.

    Notes:
    - Keeps at most ``max_entries`` in memory.
    - Evicted entries are best-effort cleaned if they expose ``close()`` or
      ``release()`` methods.
    """

    def __init__(self, max_entries: int = 8):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._items: dict[str, GPUCacheEntry] = {}
        self._order: list[str] = []

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    @staticmethod
    def _cleanup(value: Any) -> None:
        for method_name in ("close", "release"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break

    def get_or_create(
        self,
        key: str,
        factory: Callable[[], Any],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Return cached value by key, creating it with ``factory`` if absent."""
        if key in self._items:
            self._touch(key)
            return self._items[key].value

        if len(self._items) >= self.max_entries:
            evict_key = self._order.pop(0)
            evicted = self._items.pop(evict_key)
            self._cleanup(evicted.value)

        value = factory()
        self._items[key] = GPUCacheEntry(key=key, value=value, metadata=metadata or {})
        self._touch(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Return cached value and mark as recently used."""
        if key not in self._items:
            return default
        self._touch(key)
        return self._items[key].value

    def clear(self) -> None:
        """Drop all cache entries and cleanup resources."""
        for entry in self._items.values():
            self._cleanup(entry.value)
        self._items.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._items)
