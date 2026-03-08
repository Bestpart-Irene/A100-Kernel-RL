from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openenv_env.anti_hack import extract_cu_flags


DEFAULT_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
DEFAULT_NVCC_TIMEOUT_SECONDS = int(os.getenv("KERNELFORGE_NVCC_TIMEOUT", "30"))


@dataclass(slots=True)
class CompileResult:
    success: bool
    source_path: str
    output_path: str
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    extra_flags: list[str]


def _sanitize_extra_flags(extra_flags: list[str] | None) -> list[str]:
    if not extra_flags:
        return []
    pseudo_source = "\n".join(f"// CU_FLAGS: {flag}" for flag in extra_flags)
    return extract_cu_flags(pseudo_source)


def build_nvcc_command(
    source_path: str,
    output_path: str,
    cuda_source: str,
    extra_flags: list[str] | None = None,
    arch: str = DEFAULT_ARCH,
    shared: bool = True,
) -> list[str]:
    whitelisted_flags = extract_cu_flags(cuda_source)
    sanitized_extra_flags = _sanitize_extra_flags(extra_flags)
    if sanitized_extra_flags:
        whitelisted_flags = list(dict.fromkeys([*whitelisted_flags, *sanitized_extra_flags]))

    command = ["nvcc", f"-arch={arch}", "-O3", source_path, "-o", output_path]
    if shared:
        command.extend(["--shared", "-Xcompiler", "-fPIC"])
    command.extend(whitelisted_flags or ["--use_fast_math"])
    return command


def compile_cuda(
    source: str,
    extra_flags: list[str] | None = None,
    arch: str = DEFAULT_ARCH,
    output_path: str | None = None,
    timeout: int = DEFAULT_NVCC_TIMEOUT_SECONDS,
) -> CompileResult:
    workdir = Path(tempfile.mkdtemp(prefix="kernelforge_compile_"))
    src_path = workdir / "kernel.cu"
    src_path.write_text(source, encoding="utf-8")

    resolved_output = Path(output_path).resolve() if output_path else workdir / "kernel.so"
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    command = build_nvcc_command(
        source_path=str(src_path),
        output_path=str(resolved_output),
        cuda_source=source,
        extra_flags=extra_flags,
        arch=arch,
    )
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except FileNotFoundError as exc:
        returncode = -1
        stdout = ""
        stderr = str(exc)
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") or f"nvcc timed out after {timeout}s"

    if output_path and src_path.exists():
        os.unlink(src_path)

    return CompileResult(
        success=returncode == 0 and resolved_output.exists(),
        source_path=str(src_path),
        output_path=str(resolved_output),
        command=command,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        extra_flags=list(dict.fromkeys([*extract_cu_flags(source), *_sanitize_extra_flags(extra_flags)])),
    )
