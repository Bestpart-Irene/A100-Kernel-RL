"""Task routing helpers — neutral runtime module.

Moved from training/task_support.py to break the dependency inversion
where openenv_env/ imported from training/. Both openenv_env/ and
training/ now depend on this neutral module.

All functions here are re-exported from training/task_support.py
for backward compatibility.
"""
from __future__ import annotations

from training.task_support import (  # noqa: F401
    build_generation_prompt,
    build_modal_payload,
    build_prompt_lookup,
    compute_task_reward,
    evaluate_code_remote,
    filter_supported_tasks,
    infer_evaluation_backend,
    normalize_eval_result,
    normalize_task_row,
    parse_ops,
    summarize_tasks,
    support_reason,
    supports_ops6k_live_eval,
    task_interface_contract,
)
