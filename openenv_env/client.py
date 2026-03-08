"""OpenEnv client for KernelForge — used by TRL and external consumers."""
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from openenv_env.models import KernelForgeAction, KernelForgeObservation


class KernelForgeClient(EnvClient[KernelForgeAction, KernelForgeObservation, State]):
    """HTTP client for the KernelForge environment."""
    pass
