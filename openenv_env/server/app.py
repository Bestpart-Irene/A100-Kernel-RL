"""OpenEnv HTTP server for KernelForge."""
from openenv.core.env_server.http_server import create_app

from openenv_env.kernel_forge_env import KernelForgeEnv
from openenv_env.models import KernelForgeAction, KernelForgeObservation

app = create_app(
    KernelForgeEnv, KernelForgeAction, KernelForgeObservation,
    env_name="kernelforge",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
