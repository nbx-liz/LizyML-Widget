"""Subprocess entry point for OpenMP-safe job execution.

Run as: python -m lizyml_widget._subprocess_entry

Protocol:
  stdin  → pickle of {"job_type": str, "config": dict, "df_bytes": bytes,
                       "target": str, "model_out_path": str | None}
  stdout → sequence of length-prefixed pickle messages:
           {"type": "progress", "current": int, "total": int, "message": str}
           {"type": "result", "summary": dict, "eval_table": list,
            "split_summary": list, "available_plots": list}
           {"type": "error", "exc_type": str, "message": str, "traceback": str}
  SIGTERM → sets cancel flag → on_progress raises InterruptedError
"""

from __future__ import annotations

import contextlib
import pickle
import signal
import struct
import sys
import threading
import traceback
from typing import IO, Any

import pandas as pd


def send_message(output: IO[bytes], msg: dict[str, Any]) -> None:
    """Write a length-prefixed pickle message to output stream."""
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    output.write(struct.pack(">I", len(payload)))
    output.write(payload)
    output.flush()


def read_input(stream: IO[bytes]) -> dict[str, Any]:
    """Read and unpickle the input dict from stream."""
    result: dict[str, Any] = pickle.loads(stream.read())  # noqa: S301
    return result


def _create_adapter() -> Any:
    """Create a fresh LizyMLAdapter instance."""
    from lizyml_widget.adapter import LizyMLAdapter

    return LizyMLAdapter()


def run_job(
    *,
    job_type: str,
    config: dict[str, Any],
    df: pd.DataFrame,
    target: str,
    model_out_path: str | None,
    output: IO[bytes],
) -> None:
    """Execute a fit or tune job and write results to output.

    This function runs on the main thread of the subprocess,
    so OpenMP parallel regions use the correct thread pool.
    """
    cancel_flag = threading.Event()

    # Handle SIGTERM for cancellation
    def _sigterm_handler(_signum: int, _frame: Any) -> None:
        cancel_flag.set()

    with contextlib.suppress(OSError, ValueError):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    def on_progress(current: int, total: int, message: str) -> None:
        if cancel_flag.is_set():
            raise InterruptedError("Job cancelled by user")
        send_message(
            output,
            {
                "type": "progress",
                "current": current,
                "total": total,
                "message": message,
            },
        )

    try:
        adapter = _create_adapter()
        model = adapter.create_model(config, df)

        if job_type == "fit":
            summary = adapter.fit(model, on_progress=on_progress)
            result_msg: dict[str, Any] = {
                "type": "result",
                "summary": {
                    "metrics": summary.metrics,
                    "fold_count": summary.fold_count,
                    "params": summary.params,
                },
                "eval_table": adapter.evaluate_table(model),
                "split_summary": adapter.split_summary(model),
                "available_plots": adapter.available_plots(model),
                "model_path": None,
            }
        elif job_type == "tune":
            summary_t = adapter.tune(model, on_progress=on_progress)
            result_msg = {
                "type": "result",
                "tune_summary": {
                    "best_params": summary_t.best_params,
                    "best_score": summary_t.best_score,
                    "trials": summary_t.trials,
                    "metric_name": summary_t.metric_name,
                    "direction": summary_t.direction,
                },
                "eval_table": adapter.evaluate_table(model),
                "split_summary": adapter.split_summary(model),
                "available_plots": adapter.available_plots(model),
                "model_path": None,
            }
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

        # Save model if path provided
        if model_out_path:
            try:
                import shutil

                # Remove the temp dir so export() can create it fresh
                shutil.rmtree(model_out_path, ignore_errors=True)
                adapter.export_model(model, model_out_path)
                result_msg["model_path"] = model_out_path
            except Exception as save_err:
                import logging

                logging.getLogger(__name__).warning("Model save failed in subprocess: %s", save_err)

        send_message(output, result_msg)

    except Exception as exc:
        send_message(
            output,
            {
                "type": "error",
                "exc_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def main() -> None:
    """Entry point when run as python -m lizyml_widget._subprocess_entry."""
    input_data = read_input(sys.stdin.buffer)
    df = pickle.loads(input_data["df_bytes"])  # noqa: S301

    run_job(
        job_type=input_data["job_type"],
        config=input_data["config"],
        df=df,
        target=input_data["target"],
        model_out_path=input_data.get("model_out_path"),
        output=sys.stdout.buffer,
    )


if __name__ == "__main__":
    main()
