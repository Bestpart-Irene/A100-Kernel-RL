"""Anti-reward-hacking utilities for kernel evaluation.

Extends beyond forbidden-symbol scanning with runtime anti-hack checks
inspired by Dr. Kernel's three failure modes:
  1. Reward hacking (decoy kernels, no-ops)
  2. Lazy optimization (passthrough, constant output)
  3. Output shape violations

See: https://github.com/hkust-nlp/KernelGYM/blob/main/drkernel/README.md
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ALLOWED_CU_FLAGS = {
    "--use_fast_math",
    "--extra-device-vectorization",
    "--rdc=true",
}
ALLOWED_CU_FLAG_PREFIXES = ("--maxrregcount=",)

FORBIDDEN_SYMBOLS = [
    "torch",
    "at::Tensor",
    "c10::",
    "torch::autograd",
    "triton",
    "torch.compile",
    "torch.nn.functional",
]


def extract_cu_flags(cuda_code: str) -> list[str]:
    """Extract whitelisted ``// CU_FLAGS: ...`` tokens from CUDA source."""
    extracted: list[str] = []
    for line in cuda_code.splitlines():
        stripped = line.strip()
        if not stripped.startswith("// CU_FLAGS:"):
            continue

        for token in stripped.replace("// CU_FLAGS:", "").strip().split():
            if token in ALLOWED_CU_FLAGS:
                extracted.append(token)
                continue

            if any(token.startswith(prefix) for prefix in ALLOWED_CU_FLAG_PREFIXES):
                try:
                    value = int(token.split("=", 1)[1])
                except (IndexError, ValueError):
                    continue
                if 16 <= value <= 128:
                    extracted.append(token)

    # preserve order, drop duplicates
    return list(dict.fromkeys(extracted))


def scan_forbidden_symbols(so_path: str | Path) -> str | None:
    """Return a failure reason if forbidden dynamic symbols are detected."""
    so_path = str(so_path)
    try:
        proc = subprocess.run(
            ["nm", "-D", so_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        # Do not hard-fail if nm is unavailable; caller can decide fallback behavior.
        return None

    symbols = proc.stdout
    for forbidden in FORBIDDEN_SYMBOLS:
        if forbidden in symbols:
            return f"Forbidden symbol detected: {forbidden}"
    return None


# ---------------------------------------------------------------------------
# Runtime anti-hack checks (Dr. Kernel-inspired)
# ---------------------------------------------------------------------------


def check_output_not_constant(
    output1: Any,
    output2: Any,
) -> tuple[bool, str]:
    """Verify outputs differ when inputs differ.

    Catches decoy kernels that return a hardcoded constant regardless of input.
    Run the candidate kernel with two different random inputs — if the outputs
    are identical, the kernel is likely ignoring the input.

    Returns (passed, reason).
    """
    import torch

    def _flatten(val: Any) -> list:
        if isinstance(val, torch.Tensor):
            return [val]
        if isinstance(val, (list, tuple)):
            out = []
            for item in val:
                out.extend(_flatten(item))
            return out
        if isinstance(val, dict):
            out = []
            for item in val.values():
                out.extend(_flatten(item))
            return out
        return []

    tensors1 = _flatten(output1)
    tensors2 = _flatten(output2)

    if len(tensors1) != len(tensors2):
        return True, "Different output structure"

    all_identical = True
    for t1, t2 in zip(tensors1, tensors2):
        if t1.shape != t2.shape:
            return True, "Shapes differ"
        if not torch.equal(t1, t2):
            all_identical = False
            break

    if all_identical and len(tensors1) > 0:
        return False, "Output is constant across different inputs (likely decoy kernel)"

    return True, "Outputs differ for different inputs"


def check_not_noop(runtime_ms: float) -> tuple[bool, str]:
    """Flag suspiciously fast kernels that likely skip computation.

    Any real CUDA kernel launch takes at least ~2μs due to launch overhead.
    If median runtime is below 0.001ms (1μs), the kernel is almost certainly
    a no-op.

    Returns (passed, reason).
    """
    if runtime_ms < 0.001:
        return False, f"Runtime {runtime_ms:.4f}ms is below 1μs — likely a no-op kernel"
    return True, f"Runtime {runtime_ms:.3f}ms is plausible"


def check_not_passthrough(
    output: Any,
    inputs: list[Any],
) -> tuple[bool, str]:
    """Verify output is not identical to any input tensor.

    Catches lazy optimizations where the kernel just returns the input unchanged.

    Returns (passed, reason).
    """
    import torch

    def _get_first_tensor(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            return val
        if isinstance(val, (list, tuple)):
            for item in val:
                t = _get_first_tensor(item)
                if t is not None:
                    return t
        if isinstance(val, dict):
            for item in val.values():
                t = _get_first_tensor(item)
                if t is not None:
                    return t
        return None

    out_tensor = _get_first_tensor(output)
    if out_tensor is None:
        return True, "No tensor output to check"

    for i, inp in enumerate(inputs):
        inp_tensor = _get_first_tensor(inp) if not isinstance(inp, torch.Tensor) else inp
        if inp_tensor is None:
            continue
        if inp_tensor.shape == out_tensor.shape and torch.equal(inp_tensor, out_tensor):
            return False, f"Output is identical to input[{i}] — likely passthrough"

    return True, "Output differs from all inputs"


def check_shapes_match(
    candidate_output: Any,
    reference_output: Any,
) -> tuple[bool, str]:
    """Verify candidate output shapes match reference output shapes.

    Returns (passed, reason).
    """
    import torch

    def _get_shapes(val: Any) -> list[tuple[int, ...]]:
        if isinstance(val, torch.Tensor):
            return [tuple(val.shape)]
        if isinstance(val, (list, tuple)):
            out = []
            for item in val:
                out.extend(_get_shapes(item))
            return out
        if isinstance(val, dict):
            out = []
            for item in val.values():
                out.extend(_get_shapes(item))
            return out
        return []

    cand_shapes = _get_shapes(candidate_output)
    ref_shapes = _get_shapes(reference_output)

    if len(cand_shapes) != len(ref_shapes):
        return False, (
            f"Output tensor count mismatch: candidate={len(cand_shapes)}, "
            f"reference={len(ref_shapes)}"
        )

    for i, (cs, rs) in enumerate(zip(cand_shapes, ref_shapes)):
        if cs != rs:
            return False, f"Shape mismatch at output[{i}]: candidate={cs}, reference={rs}"

    return True, f"All {len(cand_shapes)} output shapes match"


def run_anti_hack_suite(
    candidate_outputs: list[Any],
    reference_outputs: list[Any],
    inputs_list: list[list[Any]],
    runtime_ms: float,
) -> tuple[bool, str]:
    """Run the full anti-hack check suite.

    Args:
        candidate_outputs: Outputs from candidate kernel on 2+ different inputs.
        reference_outputs: Outputs from reference model on same inputs.
        inputs_list: The input tensors used (2+ sets).
        runtime_ms: Median kernel runtime in milliseconds.

    Returns:
        (passed, reason) where reason explains the first failure, or "all checks passed".
    """
    # 1. Not a no-op
    passed, reason = check_not_noop(runtime_ms)
    if not passed:
        return False, reason

    # 2. Output shapes match reference
    if reference_outputs:
        passed, reason = check_shapes_match(candidate_outputs[0], reference_outputs[0])
        if not passed:
            return False, reason

    # 3. Output not constant (needs 2+ outputs)
    if len(candidate_outputs) >= 2:
        passed, reason = check_output_not_constant(
            candidate_outputs[0], candidate_outputs[1]
        )
        if not passed:
            return False, reason

    # 4. Not passthrough
    if inputs_list:
        passed, reason = check_not_passthrough(candidate_outputs[0], inputs_list[0])
        if not passed:
            return False, reason

    return True, "All anti-hack checks passed"
