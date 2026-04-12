"""Subprocess-based job runner for OpenMP-safe execution.

Spawns a fresh Python process (with optional LD_PRELOAD=libomp) to run
Fit/Tune on the subprocess main thread, avoiding libgomp pool affinity.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubprocessJobResult:
    """Result from a subprocess job execution."""

    job_type: str
    fit_summary: dict[str, Any]
    tune_summary: dict[str, Any]
    eval_table: list[dict[str, Any]]
    split_summary: list[dict[str, Any]]
    available_plots: list[str]
    model_path: str | None


def _read_exact(stream: Any, n: int) -> bytes:
    """Read exactly n bytes from stream, handling partial reads."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


_MAX_MESSAGE_BYTES = 256 * 1024 * 1024  # 256 MB safety limit


def _read_message(stream: Any) -> dict[str, Any] | None:
    """Read one length-prefixed pickle message. Returns None at EOF."""
    header = _read_exact(stream, 4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack(">I", header)
    if length > _MAX_MESSAGE_BYTES:
        raise RuntimeError(
            f"Subprocess message too large: {length} bytes (max {_MAX_MESSAGE_BYTES})"
        )
    payload = _read_exact(stream, length)
    if len(payload) < length:
        return None
    result: dict[str, Any] = pickle.loads(payload)  # noqa: S301
    return result


def run_job_subprocess(
    *,
    job_type: str,
    config: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    libomp_path: str | None,
    on_progress: Callable[..., None] | None,
    cancel_flag: threading.Event,
    model_out_path: str | None = None,
) -> SubprocessJobResult:
    """Spawn subprocess and run training job.

    Raises InterruptedError on cancellation, RuntimeError on failure.
    """
    # Serialize input
    input_data = pickle.dumps(
        {
            "job_type": job_type,
            "config": config,
            "df_bytes": pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL),
            "target": target,
            "model_out_path": model_out_path,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    # Build environment
    env = dict(os.environ)
    if libomp_path:
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{libomp_path}:{existing}" if existing else libomp_path

    # Spawn subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "lizyml_widget._subprocess_entry"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    _WAIT_TIMEOUT = 30

    try:
        # Send input and close stdin
        proc.stdin.write(input_data)  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]

        # Start cancel monitor thread
        def _cancel_monitor() -> None:
            cancel_flag.wait()
            if proc.poll() is None:
                with contextlib.suppress(OSError, ProcessLookupError):
                    proc.send_signal(signal.SIGTERM)

        monitor = threading.Thread(target=_cancel_monitor, daemon=True)
        monitor.start()

        # Read messages from stdout
        result_data: dict[str, Any] | None = None
        error_data: dict[str, Any] | None = None

        while True:
            msg = _read_message(proc.stdout)
            if msg is None:
                break

            msg_type = msg.get("type")
            if msg_type == "progress" and on_progress:
                # Forward re-tune fields (P-027) as kwargs so the parent's
                # on_progress receives the same round-aware payload it
                # would have gotten from the in-process adapter call.
                extra: dict[str, Any] = {}
                for key in (
                    "round",
                    "total_rounds",
                    "cumulative_trials",
                    "expanded_dims",
                    "latest_score",
                    "latest_state",
                    "best_score",
                ):
                    if key in msg:
                        extra[key] = msg[key]
                on_progress(msg["current"], msg["total"], msg["message"], **extra)
            elif msg_type == "result":
                result_data = msg
            elif msg_type == "error":
                error_data = msg

        try:
            proc.wait(timeout=_WAIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(
                f"Subprocess did not exit within {_WAIT_TIMEOUT}s after job completion"
            ) from None

    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise

    finally:
        # Ensure cancel monitor terminates and clean up pipes
        cancel_flag.set()
        monitor.join(timeout=2.0)
        with contextlib.suppress(Exception):
            proc.stdout.close()  # type: ignore[union-attr]
        with contextlib.suppress(Exception):
            proc.stderr.close()  # type: ignore[union-attr]

    # Handle error from subprocess
    if error_data:
        exc_type = error_data.get("exc_type", "RuntimeError")
        message = error_data.get("message", "Unknown error in subprocess")
        if exc_type == "InterruptedError":
            raise InterruptedError(message)
        raise RuntimeError(f"[{exc_type}] {message}")

    # Handle missing result
    if result_data is None:
        stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""  # type: ignore[union-attr]
        raise RuntimeError(
            f"Subprocess exited with code {proc.returncode} "
            f"without result. stderr: {stderr_output[:500]}"
        )

    return SubprocessJobResult(
        job_type=job_type,
        fit_summary=result_data.get("summary", {}),
        tune_summary=result_data.get("tune_summary", {}),
        eval_table=result_data.get("eval_table", []),
        split_summary=result_data.get("split_summary", []),
        available_plots=result_data.get("available_plots", []),
        model_path=result_data.get("model_path"),
    )
