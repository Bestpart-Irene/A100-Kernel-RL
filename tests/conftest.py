"""Pytest configuration — mock OpenEnv SDK when not installed.

The openenv-core SDK is only available inside OpenEnv Docker containers
or on Modal. For local testing, we mock the SDK's base classes so that
kernel_forge_env.py can be imported and tested with mocked Modal calls.

Uses sys.modules.setdefault() so the real SDK takes precedence when available.
"""
import sys
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict, Field


# --- Mock OpenEnv SDK base classes ---

class _MockAction(BaseModel):
    """Stand-in for openenv.core.env_server.types.Action."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: dict = Field(default_factory=dict)


class _MockObservation(BaseModel):
    """Stand-in for openenv.core.env_server.types.Observation."""
    reward: float = 0.0
    done: bool = False
    metadata: dict = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _MockState(BaseModel):
    """Stand-in for openenv.core.env_server.types.State."""
    model_config = ConfigDict(extra="allow")
    episode_id: str | None = None
    step_count: int = 0


class _MockEnvironment:
    """Stand-in for openenv.core.env_server.Environment."""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        pass


# --- Patch sys.modules before any test imports kernel_forge_env ---

_env_server = MagicMock()
_env_server.Environment = _MockEnvironment
_env_server.create_fastapi_app = MagicMock(return_value=MagicMock())

_types = MagicMock()
_types.Action = _MockAction
_types.Observation = _MockObservation
_types.State = _MockState

_core = MagicMock()
_core.env_server = _env_server
_core.env_server.types = _types

_openenv = MagicMock()
_openenv.core = _core

sys.modules.setdefault("openenv", _openenv)
sys.modules.setdefault("openenv.core", _core)
sys.modules.setdefault("openenv.core.env_server", _env_server)
sys.modules.setdefault("openenv.core.env_server.types", _types)

_http_server = MagicMock()
_http_server.create_app = MagicMock(return_value=MagicMock())
sys.modules.setdefault("openenv.core.env_server.http_server", _http_server)

_env_client = MagicMock()
sys.modules.setdefault("openenv.core.env_client", _env_client)
