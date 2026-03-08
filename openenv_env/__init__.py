"""
KernelForge OpenEnv Environment for target-GPU CUDA kernel RL training.

OpenEnv is an independent framework from Meta-PyTorch (NOT a Gymnasium
extension). HTTP client-server architecture with Docker container isolation.
Correct install (uv): uv add "openenv-core[core]>=0.2.1" (NOT "openenv")
"""
__all__ = ["KernelForgeEnv", "KernelForgeAction", "KernelForgeObservation"]


def __getattr__(name: str):
    if name in ("KernelForgeAction", "KernelForgeObservation"):
        from openenv_env.models import KernelForgeAction, KernelForgeObservation
        return KernelForgeAction if name == "KernelForgeAction" else KernelForgeObservation
    if name == "KernelForgeEnv":
        from openenv_env.kernel_forge_env import KernelForgeEnv
        return KernelForgeEnv
    if name == "KernelForgeClient":
        from openenv_env.client import KernelForgeClient
        return KernelForgeClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
