"""WidgetActionDispatcher — routes JS action payloads to handlers.

Extracted from ``widget.py`` per #127 to keep the widget module under the
800-line ceiling declared in CLAUDE.md §8. The dispatcher owns no
traitlets — it holds a reference to the parent ``LizyWidget`` and reads
/ writes its traitlets through that handle.

State-machine transitions still live in ``widget.py::_supervise`` /
``_run_job``; this module is purely the action-dispatch boundary.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from .types import ConfigPatchOp

if TYPE_CHECKING:
    from .widget import LizyWidget

_log = logging.getLogger(__name__)

# ── Class-level constants (frozen sets / regexes) ────────────────────

_VALID_COL_TYPES: frozenset[str] = frozenset({"numeric", "categorical"})

#: Safe path pattern for ``patch_config``: dotted identifiers, no dunder.
_SAFE_PATH_RE = re.compile(r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*$")
_VALID_PATCH_OPS: frozenset[str] = frozenset({"set", "unset", "merge"})

#: Upper bound on retune n_trials so a tampered UI payload cannot
#: overwhelm the backend.  Matches the NumericStepper ``max`` in
#: RetuneControls.tsx; values outside the range are silently dropped.
_RETUNE_MAX_N_TRIALS = 10_000

#: Binary buffer threshold for large Plotly JSON (800 KB) — D-1.
_PLOT_BINARY_THRESHOLD = 800_000


class WidgetActionDispatcher:
    """Routes JS action payloads to typed handlers on the widget.

    The dispatcher is constructed with a reference to the ``LizyWidget``
    instance and reads/writes its traitlets through that handle. Each
    handler reproduces the behaviour of the previous ``LizyWidget._handle_*``
    method exactly; this is a mechanical extraction with no semantic change.

    Unknown action types are silently ignored, matching the previous
    contract — JS may send forward-compatible action names without the
    widget breaking.
    """

    def __init__(self, widget: LizyWidget) -> None:
        self._widget = widget
        self._service = widget._service

    # ── Entry points ──────────────────────────────────────────────

    def dispatch(self, action_type: str, payload: dict[str, Any]) -> None:
        """Look up the handler for *action_type* and invoke it.

        Unknown action types return without effect (forward compatibility).
        """
        handler = self._HANDLERS.get(action_type)
        if handler is None:
            return
        handler(self, payload)

    # ── Data tab handlers ─────────────────────────────────────────

    def handle_set_target(self, payload: dict[str, Any]) -> None:
        target = payload.get("target", "")
        if not target:
            return
        w = self._widget
        try:
            df_info = self._service.set_target(target)
            w.df_info = df_info
            w.status = "data_loaded"
            task = df_info.get("task")
            if task:
                w.config = self._service.apply_task_params(dict(w.config), task)
        except Exception as e:
            w.error = {"code": "TARGET_ERROR", "message": str(e)}

    def handle_set_task(self, payload: dict[str, Any]) -> None:
        task = payload.get("task", "")
        if not task:
            return
        w = self._widget
        try:
            df_info = self._service.set_task(task)
            w.df_info = df_info
            w.config = self._service.apply_task_params(dict(w.config), task)
        except Exception as e:
            w.error = {"code": "TASK_ERROR", "message": str(e)}

    def handle_update_column(self, payload: dict[str, Any]) -> None:
        w = self._widget
        name = payload.get("name")
        if not name:
            w.error = {"code": "COLUMN_ERROR", "message": "Missing column name"}
            return
        col_type = payload.get("col_type", "numeric")
        if col_type not in _VALID_COL_TYPES:
            w.error = {"code": "COLUMN_ERROR", "message": f"Invalid col_type: {col_type!r}"}
            return
        try:
            df_info = self._service.update_column(
                name,
                excluded=payload.get("excluded", False),
                col_type=col_type,
            )
            w.df_info = df_info
        except Exception as e:
            w.error = {"code": "COLUMN_ERROR", "message": str(e)}

    def handle_update_cv(self, payload: dict[str, Any]) -> None:
        w = self._widget
        strategy = payload.get("strategy", "kfold")
        # #130: the widget no longer keeps a hardcoded fallback list of CV
        # strategies. The backend contract is the single source of truth;
        # if it is missing or incomplete the action fails fast so a backend
        # that adds a new CV strategy is never silently rejected against an
        # outdated allowlist.
        valid = w.backend_contract.get("capabilities", {}).get("cv_strategies")
        if not valid:
            w.error = {
                "code": "BACKEND_NOT_READY",
                "message": (
                    "Backend contract is not loaded yet. Call load(df) before configuring CV."
                ),
            }
            return
        valid_set = frozenset(valid)
        if strategy not in valid_set:
            w.error = {"code": "CV_ERROR", "message": f"Invalid strategy: {strategy!r}"}
            return
        try:
            n_splits = int(payload.get("n_splits", 5))
        except (ValueError, TypeError) as e:
            w.error = {"code": "CV_ERROR", "message": f"Invalid n_splits: {e}"}
            return
        if not (2 <= n_splits <= 100):
            w.error = {"code": "CV_ERROR", "message": f"n_splits must be 2-100, got {n_splits}"}
            return
        try:
            df_info = self._service.update_cv(
                strategy,
                n_splits,
                group_column=payload.get("group_column"),
                time_column=payload.get("time_column"),
                random_state=payload.get("random_state", 42),
                shuffle=payload.get("shuffle", True),
                gap=payload.get("gap", 0),
                purge_gap=payload.get("purge_gap", 0),
                embargo=payload.get("embargo", 0),
                train_size_max=payload.get("train_size_max"),
                test_size_max=payload.get("test_size_max"),
                blocks=payload.get("blocks"),
                groups=payload.get("groups"),
                min_train_rows=int(payload.get("min_train_rows", 0)),
                min_valid_rows=int(payload.get("min_valid_rows", 0)),
            )
            w.df_info = df_info
        except Exception as e:
            w.error = {"code": "CV_ERROR", "message": str(e)}

    def handle_get_column_stats(self, payload: dict[str, Any]) -> None:
        w = self._widget
        column = payload.get("column", "")
        if not column:
            w.send({"type": "column_stats_error", "message": "Missing column name"})
            return
        try:
            result = self._service.get_column_stats(column)
            w.send(
                {
                    "type": "column_stats",
                    "column": result["column"],
                    "unique_count": result["unique_count"],
                    "dtype": result["dtype"],
                    "values": result["values"],
                    "truncated": result.get("truncated", False),
                }
            )
        except Exception as e:
            w.send({"type": "column_stats_error", "message": str(e)})

    def handle_preview_splits(self, _payload: dict[str, Any]) -> None:
        w = self._widget
        try:
            result = self._service.preview_splits()
            w.send({"type": "preview_splits", **result})
        except Exception as e:
            w.send({"type": "preview_splits_error", "message": str(e)})

    # ── Config tab handlers ───────────────────────────────────────

    def handle_patch_config(self, payload: dict[str, Any]) -> None:
        w = self._widget
        raw_ops = payload.get("ops", [])
        if not raw_ops:
            return
        ops: list[ConfigPatchOp] = []
        for o in raw_ops:
            path = o.get("path", "")
            if not _SAFE_PATH_RE.match(path) or "__" in path:
                w.error = {"code": "INVALID_PATCH", "message": f"Invalid patch path: {path!r}"}
                return
            op_str = o.get("op", "")
            if op_str not in _VALID_PATCH_OPS:
                w.error = {"code": "INVALID_PATCH", "message": f"Invalid op: {op_str!r}"}
                return
            ops.append(ConfigPatchOp(op=op_str, path=path, value=o.get("value")))
        try:
            w.config = self._service.apply_config_patch(dict(w.config), ops)
        except Exception as e:
            w.error = {"code": "PATCH_ERROR", "message": str(e)}

    def handle_import_yaml(self, payload: dict[str, Any]) -> None:
        w = self._widget
        content = payload.get("content", "")
        if not content:
            return
        try:
            import yaml  # type: ignore[import-untyped]

            loaded: dict[str, Any] = yaml.safe_load(content)
            if not isinstance(loaded, dict):
                w.error = {"code": "IMPORT_ERROR", "message": "Invalid YAML content"}
                return
            w._apply_loaded_config(loaded)
        except Exception as e:
            w.error = {"code": "IMPORT_ERROR", "message": str(e)}

    def handle_export_yaml(self, _payload: dict[str, Any]) -> None:
        w = self._widget
        try:
            import yaml  # type: ignore[import-untyped]

            full_config = self._service.build_config(dict(w.config))
            content = yaml.dump(full_config, default_flow_style=False)
            w.send({"type": "yaml_export", "content": content})
        except Exception as e:
            w.error = {"code": "EXPORT_ERROR", "message": str(e)}

    def handle_raw_config(self, _payload: dict[str, Any]) -> None:
        w = self._widget
        try:
            import yaml  # type: ignore[import-untyped]

            if self._service.has_data() and self._service.has_target():
                full_config = self._service.build_config(dict(w.config))
            else:
                full_config = dict(w.config)
            content = yaml.dump(full_config, default_flow_style=False)
            w.send({"type": "raw_config", "content": content})
        except Exception as e:
            with contextlib.suppress(Exception):
                w.send({"type": "raw_config_error", "message": str(e)})
            w.error = {"code": "EXPORT_ERROR", "message": str(e)}

    # ── Job handlers ──────────────────────────────────────────────

    def handle_fit(self, _payload: dict[str, Any]) -> None:
        self._widget._run_job("fit")

    def handle_tune(self, _payload: dict[str, Any]) -> None:
        self._widget._run_job("tune")

    def handle_retune(self, payload: dict[str, Any]) -> None:
        """Dispatch a resume-style tune run (P-028).

        Refuses to run when no prior tune exists (synchronous error: no
        job thread is spawned).  Payload fields are validated here so the
        worker thread only ever sees a trusted dict.
        """
        w = self._widget
        if not w.tune_summary:
            w.error = {
                "code": "NO_PRIOR_TUNE",
                "message": ("Cannot retune: no prior tune result on this widget. Run Tune first."),
            }
            w.status = "failed"
            return
        kwargs: dict[str, Any] = {"resume": True}
        n_trials = payload.get("n_trials")
        # ``bool`` is a subclass of ``int``; exclude it explicitly so that
        # ``{"n_trials": True}`` doesn't sneak past the bounds check.
        if (
            isinstance(n_trials, int)
            and not isinstance(n_trials, bool)
            and 0 < n_trials <= _RETUNE_MAX_N_TRIALS
        ):
            kwargs["n_trials"] = n_trials
        elif n_trials is not None:
            _log.warning(
                "retune action: rejecting invalid n_trials=%r (expected int in 1..%d); "
                "falling back to backend default",
                n_trials,
                _RETUNE_MAX_N_TRIALS,
            )
        expand_boundary = payload.get("expand_boundary")
        if isinstance(expand_boundary, bool):
            kwargs["expand_boundary"] = expand_boundary
        elif expand_boundary is not None:
            _log.warning(
                "retune action: rejecting invalid expand_boundary=%r (expected bool); "
                "falling back to backend default",
                expand_boundary,
            )
        boundary_threshold = payload.get("boundary_threshold")
        if (
            isinstance(boundary_threshold, (int, float))
            and not isinstance(boundary_threshold, bool)
            and 0.0 <= boundary_threshold <= 1.0
        ):
            kwargs["boundary_threshold"] = float(boundary_threshold)
        else:
            if boundary_threshold is not None:
                _log.warning(
                    "retune action: rejecting invalid boundary_threshold=%r "
                    "(expected float in [0.0, 1.0]); using default 0.05",
                    boundary_threshold,
                )
            kwargs["boundary_threshold"] = 0.05
        w._run_job("tune", retune_kwargs=kwargs)

    def handle_cancel(self, _payload: dict[str, Any]) -> None:
        self._widget._cancel_flag.set()

    # ── Plot helpers ──────────────────────────────────────────────

    def _send_plot_response(
        self,
        plot_type: str,
        plotly_json: str,
        request_id: str | None,
    ) -> None:
        """Send a plot_data response, using a binary buffer for large payloads.

        When *plotly_json* exceeds ``_PLOT_BINARY_THRESHOLD`` bytes, the JSON
        string is sent as a binary buffer instead of inline in the message dict.
        The JS side detects this via the ``binary`` flag and decodes the buffer.
        """
        msg: dict[str, Any] = {
            "type": "plot_data",
            "plot_type": plot_type,
        }
        if request_id is not None:
            msg["request_id"] = request_id

        if len(plotly_json) > _PLOT_BINARY_THRESHOLD:
            msg["binary"] = True
            self._widget.send(msg, buffers=[plotly_json.encode("utf-8")])
        else:
            msg["plotly_json"] = plotly_json
            self._widget.send(msg)

    def handle_request_plot(self, payload: dict[str, Any]) -> None:
        # TODO(C-4): Consider non-blocking plot generation for slow plots
        # (e.g., SHAP). Deferred because self.send() from a background thread
        # is unreliable on Colab (BG thread comm blackout), and routing the
        # response back to the main thread adds complexity.  Re-evaluate when
        # a main-thread callback mechanism is available.
        w = self._widget
        plot_type = payload.get("plot_type", "")
        if not plot_type:
            return
        raw_rid = payload.get("request_id")
        request_id: str | None = raw_rid if isinstance(raw_rid, str) else None
        # Whitelist allowed keys and validate types to prevent untrusted input
        options = payload.get("options")
        kwargs: dict[str, Any] = {}
        if isinstance(options, dict):
            metrics = options.get("metrics")
            if isinstance(metrics, list) and all(isinstance(m, str) for m in metrics):
                kwargs["metrics"] = metrics
        try:
            plot_data = self._service.get_plot(plot_type, **kwargs)
            self._send_plot_response(plot_type, plot_data.plotly_json, request_id)
        except Exception as e:
            err: dict[str, Any] = {
                "type": "plot_error",
                "plot_type": plot_type,
                "message": str(e),
            }
            if request_id is not None:
                err["request_id"] = request_id
            w.send(err)

    def handle_request_inference_plot(self, payload: dict[str, Any]) -> None:
        w = self._widget
        plot_type = payload.get("plot_type", "")
        if not plot_type:
            return
        raw_rid = payload.get("request_id")
        request_id: str | None = raw_rid if isinstance(raw_rid, str) else None
        # Inference plots use prediction data, not fit model
        inference_data = w.inference_result.get("data", [])
        if not inference_data:
            self.handle_request_plot(payload)
            return
        try:
            predictions = pd.DataFrame(inference_data)
            plot_data = self._service.get_inference_plot(predictions, plot_type)
            self._send_plot_response(plot_type, plot_data.plotly_json, request_id)
        except Exception as exc:
            _log.debug("Inference plot failed, falling back to fit plot: %s", exc)
            self.handle_request_plot(payload)

    # ── Inference handlers ────────────────────────────────────────

    def handle_run_inference(self, payload: dict[str, Any]) -> None:
        w = self._widget
        if w._inference_df is None:
            w.error = {"code": "INFERENCE_ERROR", "message": "No inference data loaded"}
            return
        try:
            return_shap = payload.get("return_shap", False)
            result = self._service.predict(w._inference_df, return_shap=return_shap)
            records = result.predictions.to_dict(orient="records")
            w.inference_result = {
                "status": "completed",
                "rows": len(records),
                "data": records,
                "warnings": result.warnings,
            }
        except Exception as e:
            w.inference_result = {
                "status": "failed",
                "message": str(e),
            }

    def handle_apply_best_params(self, payload: dict[str, Any]) -> None:
        # P-035: snapshots are owned by Service via _last_tune_summary;
        # widget no longer holds _tune_config_snapshot / _tune_ui_snapshot.
        w = self._widget
        params = payload.get("params", {})
        if not params:
            return
        try:
            w.config = self._service.apply_best_params(params, dict(w.config))
        except Exception as e:
            w.error = {"code": "APPLY_ERROR", "message": str(e)}

    def handle_export_code(self, _payload: dict[str, Any]) -> None:
        w = self._widget
        try:
            import shutil
            import tempfile
            from pathlib import Path

            result_path = self._service.export_code(None)  # always tmpdir for UI
            zip_dir = tempfile.mkdtemp(prefix="lzw_code_export_")
            zip_base = str(Path(zip_dir) / "exported_code")
            zip_path = shutil.make_archive(zip_base, "zip", str(result_path))

            with open(zip_path, "rb") as f:
                zip_bytes = f.read()

            w.send(
                {"type": "code_export_download", "filename": "exported_code.zip"},
                buffers=[zip_bytes],
            )

            # Cleanup temp files
            with contextlib.suppress(OSError):
                os.unlink(zip_path)
            with contextlib.suppress(OSError):
                shutil.rmtree(str(result_path), ignore_errors=True)
            with contextlib.suppress(OSError):
                shutil.rmtree(zip_dir, ignore_errors=True)
        except Exception as e:
            w.error = {"code": "EXPORT_CODE_ERROR", "message": str(e)}

    # ── Handler registry ──────────────────────────────────────────

    _HANDLERS: dict[str, Callable[[WidgetActionDispatcher, dict[str, Any]], None]] = {
        "set_target": handle_set_target,
        "set_task": handle_set_task,
        "update_column": handle_update_column,
        "update_cv": handle_update_cv,
        "get_column_stats": handle_get_column_stats,
        "preview_splits": handle_preview_splits,
        "patch_config": handle_patch_config,
        "fit": handle_fit,
        "tune": handle_tune,
        "retune": handle_retune,
        "cancel": handle_cancel,
        "request_plot": handle_request_plot,
        "run_inference": handle_run_inference,
        "apply_best_params": handle_apply_best_params,
        "request_inference_plot": handle_request_inference_plot,
        "import_yaml": handle_import_yaml,
        "export_yaml": handle_export_yaml,
        "raw_config": handle_raw_config,
        "export_code": handle_export_code,
    }
