"""Pydantic models for KernelForge OpenEnv environment."""
from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class KernelForgeAction(Action):
    """Action payload: CUDA kernel source code."""

    cuda_code: str = Field(..., description="CUDA kernel source code string")


class KernelForgeObservation(Observation):
    """Observation payload returned by KernelForgeEnv."""

    text: str = Field(..., description="Environment feedback text")
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any] = Field(default_factory=dict)
    graph_properties: dict[str, Any] | None = Field(
        default=None,
        description="Graph topology properties (degree dist, density, diameter, etc.)",
    )
    topology_type: str | None = Field(
        default=None,
        description="Graph topology class: power-law, sparse-islands, dense-regular, etc.",
    )
