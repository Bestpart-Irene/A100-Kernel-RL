"""
FastAPI eval service for CoreWeave/Northflank deployment.

Exposes the same 5 evaluation endpoints as the Modal app, but served as
a persistent HTTP service on a CoreWeave A100 GPU via Northflank.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from eval_service.eval_core import (
    evaluate_kernel_impl,
    evaluate_kernels_batch_impl,
    evaluate_ops6k_kernel_impl,
    profile_baselines_impl,
    test_gpu_features_impl,
)

app = FastAPI(
    title="KernelForge Eval Service",
    description="CUDA kernel evaluation on CoreWeave A100 via Northflank",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Liveness probe for Northflank."""
    return {"status": "ok"}


@app.post("/evaluate_kernel")
def evaluate_kernel(payload: dict):
    """Compile, verify (PAC), and benchmark a WCC CUDA kernel."""
    return evaluate_kernel_impl(payload)


@app.post("/evaluate_ops6k_kernel")
def evaluate_ops6k_kernel(payload: dict):
    """Evaluate a CUDA kernel against a PyTorch reference from Ops-6K."""
    return evaluate_ops6k_kernel_impl(payload)


@app.post("/evaluate_kernels_batch")
def evaluate_kernels_batch(payloads: list[dict]):
    """Evaluate multiple kernels in a single call."""
    return evaluate_kernels_batch_impl(payloads)


@app.post("/profile_baselines")
def profile_baselines(payload: dict | None = None):
    """Profile baseline kernels on the target GPU."""
    return profile_baselines_impl()


@app.post("/test_gpu_features")
def test_gpu_features(payload: dict | None = None):
    """Test GPU feature availability."""
    return test_gpu_features_impl()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return eval-compatible error responses instead of 500s."""
    return JSONResponse(
        status_code=200,
        content={
            "compiles": False,
            "correct": False,
            "error": f"Eval service error: {str(exc)[:1000]}",
            "runtime_ms": 0.0,
            "runtime_stats": {},
            "speedup_vs_orig": 0.0,
            "speedup_vs_dg": 0.0,
        },
    )
