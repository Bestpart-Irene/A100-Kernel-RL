from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    crashed: bool
    duration_seconds: float
    command: list[str]
    script_path: str


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except Exception:
        return f"SIG{-returncode}"


def run_in_sandbox(script: str, timeout: int = 30) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="kernelforge_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "script.py"
        script_path.write_text(script, encoding="utf-8")
        command = [sys.executable, str(script_path)]
        started = time.monotonic()

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired as exc:
            duration_seconds = time.monotonic() - started
            stderr = (exc.stderr or "").strip()
            if not stderr:
                stderr = f"Process exceeded timeout of {timeout}s"
            return SandboxResult(
                stdout=(exc.stdout or ""),
                stderr=stderr,
                returncode=-1,
                timed_out=True,
                crashed=False,
                duration_seconds=duration_seconds,
                command=command,
                script_path=str(script_path),
            )

        duration_seconds = time.monotonic() - started
        crashed = proc.returncode < 0
        stderr = proc.stderr or ""
        if crashed and not stderr:
            stderr = f"Process crashed with signal {_signal_name(proc.returncode)}"

        return SandboxResult(
            stdout=proc.stdout,
            stderr=stderr,
            returncode=proc.returncode,
            timed_out=False,
            crashed=crashed,
            duration_seconds=duration_seconds,
            command=command,
            script_path=str(script_path),
        )
