from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openenv_env.anti_hack import scan_forbidden_symbols
from verification.pac_verify import generate_test_graphs, run_kernel_verification, verify_wcc


@dataclass(slots=True)
class VerifyResult:
    correct: bool
    verifier_msg: str
    graphs_checked: int
    forbidden_symbol_error: str | None
    metadata: dict[str, Any]


def verify_kernel(
    so_path: str,
    task_code: str = "",
    num_graphs: int = 5,
    num_vertices: int = 10000,
) -> VerifyResult:
    library_path = Path(so_path)
    if not library_path.exists():
        return VerifyResult(
            correct=False,
            verifier_msg=f"Compiled library not found: {library_path}",
            graphs_checked=0,
            forbidden_symbol_error=None,
            metadata={"task_code_present": bool(task_code.strip())},
        )

    forbidden_symbol_error = scan_forbidden_symbols(library_path)
    if forbidden_symbol_error:
        return VerifyResult(
            correct=False,
            verifier_msg=forbidden_symbol_error,
            graphs_checked=0,
            forbidden_symbol_error=forbidden_symbol_error,
            metadata={"task_code_present": bool(task_code.strip())},
        )

    graphs = generate_test_graphs(num_vertices=num_vertices)[:num_graphs]
    for checked_count, (graph_type, edges, n_verts) in enumerate(graphs, start=1):
        try:
            labels = run_kernel_verification(str(library_path), edges, n_verts)
            passed, message = verify_wcc(labels, edges, n_verts)
        except Exception as exc:
            return VerifyResult(
                correct=False,
                verifier_msg=f"Verification exception on {graph_type}: {str(exc)[:500]}",
                graphs_checked=checked_count - 1,
                forbidden_symbol_error=None,
                metadata={
                    "task_code_present": bool(task_code.strip()),
                    "graph_type": graph_type,
                },
            )

        if not passed:
            return VerifyResult(
                correct=False,
                verifier_msg=f"FAILED on {graph_type}: {message}",
                graphs_checked=checked_count,
                forbidden_symbol_error=None,
                metadata={
                    "task_code_present": bool(task_code.strip()),
                    "graph_type": graph_type,
                },
            )

    return VerifyResult(
        correct=True,
        verifier_msg=f"All {len(graphs)} graphs verified",
        graphs_checked=len(graphs),
        forbidden_symbol_error=None,
        metadata={"task_code_present": bool(task_code.strip())},
    )
