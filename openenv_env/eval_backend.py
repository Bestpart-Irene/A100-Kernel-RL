"""
Eval dispatch abstraction — routes to CoreWeave (HTTP) or Modal (serverless).

Set KERNELFORGE_EVAL_BACKEND to control dispatch:
  - "coreweave" (default): HTTP POST to KERNELFORGE_EVAL_URL (Northflank endpoint)
  - "modal": modal.Function.from_name().remote() (requires Modal auth)
"""
from __future__ import annotations

import os
from typing import Any

EVAL_BACKEND = os.getenv("KERNELFORGE_EVAL_BACKEND", "coreweave")
EVAL_URL = os.getenv("KERNELFORGE_EVAL_URL", "")
MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")


def dispatch_eval(fn_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch an evaluation call to the configured backend.

    Args:
        fn_name: Evaluation function name (e.g. "evaluate_kernel", "profile_baselines").
        payload: JSON-serializable payload for the evaluation function.

    Returns:
        Evaluation result dict.
    """
    if EVAL_BACKEND == "modal":
        return _dispatch_modal(fn_name, payload)
    return _dispatch_http(fn_name, payload)


def _dispatch_http(fn_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch via HTTP POST to CoreWeave/Northflank eval service."""
    import httpx

    if not EVAL_URL:
        raise RuntimeError(
            "KERNELFORGE_EVAL_URL must be set when KERNELFORGE_EVAL_BACKEND=coreweave. "
            "Set it to the Northflank eval service URL (e.g. https://eval-kernelforge.northflank.app)."
        )
    url = f"{EVAL_URL.rstrip('/')}/{fn_name}"
    resp = httpx.post(url, json=payload or {}, timeout=300.0)
    resp.raise_for_status()
    return resp.json()


def _dispatch_modal(fn_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch via Modal serverless function."""
    import modal

    fn = modal.Function.from_name(MODAL_APP_NAME, fn_name)
    if payload is None:
        return fn.remote()
    return fn.remote(payload)
