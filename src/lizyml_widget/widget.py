"""LizyWidget - anywidget-based notebook UI for LizyML."""

from __future__ import annotations

import contextlib
import copy
import importlib.resources
import statistics
import threading
import time
import traceback
from typing import Any

import anywidget
import pandas as pd
import traitlets

from .adapter import BackendAdapter, LizyMLAdapter
from .service import WidgetService
from .types import ConfigPatchOp, FitSummary, PredictionSummary, TuningSummary


class LizyWidget(anywidget.AnyWidget):
    """LizyML notebook UI widget."""

    _esm = importlib.resources.files("lizyml_widget") / "static/widget.js"
    _css = importlib.resources.files("lizyml_widget") / "static/widget.css"

    # ── Python → JS traitlets ────────────────────────────────
    backend_info = traitlets.Dict({}).tag(sync=True)
    df_info = traitlets.Dict({}).tag(sync=True)
    backend_contract = traitlets.Dict({}).tag(sync=True)
    config = traitlets.Dict({}).tag(sync=True)
    status = traitlets.Unicode("idle").tag(sync=True)
    job_type = traitlets.Unicode("").tag(sync=True)
    job_index = traitlets.Int(0).tag(sync=True)
    progress = traitlets.Dict({}).tag(sync=True)
    elapsed_sec = traitlets.Float(0.0).tag(sync=True)
    fit_summary = traitlets.Dict({}).tag(sync=True)
    tune_summary = traitlets.Dict({}).tag(sync=True)
    available_plots: list[str] = traitlets.List([]).tag(sync=True)  # type: ignore[assignment]
    inference_result = traitlets.Dict({}).tag(sync=True)
    error = traitlets.Dict({}).tag(sync=True)

    # ── JS → Python traitlet ─────────────────────────────────
    action = traitlets.Dict({}).tag(sync=True)

    def __init__(self, *, adapter: BackendAdapter | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._service = WidgetService(adapter=adapter or LizyMLAdapter())
        self._job_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._job_counter = 0
        self._inference_df: pd.DataFrame | None = None
        self._tune_config_snapshot: dict[str, Any] | None = None

        info = self._service.info
        self.backend_info = {"name": info.name, "version": info.version}

    # ── Public Python API ─────────────────────────────────────

    def load(self, df: pd.DataFrame, target: str | None = None) -> LizyWidget:
        """Load a DataFrame into the widget."""
        df_info = self._service.load_data(df, target=target)
        self.df_info = df_info
        self.status = "data_loaded"
        self.error = {}
        self.fit_summary = {}
        self.tune_summary = {}
        self.available_plots = []

        contract = self._service.get_backend_contract()
        self.backend_contract = {
            "schema_version": contract.schema_version,
            "config_schema": contract.config_schema,
            "ui_schema": contract.ui_schema,
            "capabilities": contract.capabilities,
        }
        self.config = self._service.initialize_config()
        return self

    def set_target(self, col: str) -> LizyWidget:
        """Set target column and trigger auto-detection."""
        df_info = self._service.set_target(col)
        self.df_info = df_info
        self.status = "data_loaded"
        return self

    def fit(self, *, timeout: float | None = None) -> LizyWidget:
        """Run Fit in a background thread and block until complete.

        Raises RuntimeError if the job fails.
        """
        done = threading.Event()

        def _watch(change: dict[str, Any]) -> None:
            if change["new"] in ("completed", "failed"):
                done.set()

        self.observe(_watch, names=["status"])
        self._run_job("fit")
        # Validation failure sets status synchronously before thread starts
        if self.status in ("completed", "failed"):
            done.set()
        done.wait(timeout=timeout)
        self.unobserve(_watch, names=["status"])
        if self.status == "failed":
            msg = self.error.get("message", "Fit failed")
            raise RuntimeError(msg)
        return self

    def tune(self, *, timeout: float | None = None) -> LizyWidget:
        """Run Tune in a background thread and block until complete.

        Raises RuntimeError if the job fails.
        """
        done = threading.Event()

        def _watch(change: dict[str, Any]) -> None:
            if change["new"] in ("completed", "failed"):
                done.set()

        self.observe(_watch, names=["status"])
        self._run_job("tune")
        if self.status in ("completed", "failed"):
            done.set()
        done.wait(timeout=timeout)
        self.unobserve(_watch, names=["status"])
        if self.status == "failed":
            msg = self.error.get("message", "Tune failed")
            raise RuntimeError(msg)
        return self

    @property
    def task(self) -> str | None:
        """Auto-detected task type (binary / multiclass / regression)."""
        return self.df_info.get("task")

    @property
    def cv_method(self) -> str:
        """Current CV strategy name."""
        return str(self.df_info.get("cv", {}).get("strategy", "kfold"))

    @property
    def cv_n_splits(self) -> int:
        """Number of CV splits."""
        return int(self.df_info.get("cv", {}).get("n_splits", 5))

    @property
    def df_shape(self) -> list[int]:
        """Shape [rows, cols] of the loaded DataFrame."""
        return list(self.df_info.get("shape", []))

    @property
    def df_columns(self) -> list[dict[str, Any]]:
        """Column metadata list from the loaded DataFrame."""
        return list(self.df_info.get("columns", []))

    def load_inference(self, df: pd.DataFrame) -> LizyWidget:
        """Load a DataFrame for inference."""
        self._inference_df = df
        self.inference_result = {"status": "ready", "rows": len(df)}
        return self

    def set_config(self, config: dict[str, Any]) -> LizyWidget:
        """Set config programmatically. Canonicalizes via adapter defaults."""
        if "config_version" not in config:
            existing_version = self.config.get("config_version", 1)
            config = {**config, "config_version": existing_version}
        self.config = self._service.canonicalize_config(config)
        return self

    def get_config(self) -> dict[str, Any]:
        """Get current config."""
        return dict(self.config)

    def get_fit_summary(self) -> FitSummary | None:
        """Get last fit summary."""
        if not self.fit_summary:
            return None
        return FitSummary(
            metrics=self.fit_summary["metrics"],
            fold_count=self.fit_summary["fold_count"],
            params=self.fit_summary["params"],
        )

    def get_tune_summary(self) -> TuningSummary | None:
        """Get last tune summary."""
        if not self.tune_summary:
            return None
        return TuningSummary(
            best_params=self.tune_summary["best_params"],
            best_score=self.tune_summary["best_score"],
            trials=self.tune_summary["trials"],
            metric_name=self.tune_summary["metric_name"],
            direction=self.tune_summary["direction"],
        )

    def get_model(self) -> Any:
        """Get the underlying trained model object."""
        return self._service.get_model()

    def predict(self, df: pd.DataFrame, *, return_shap: bool = False) -> PredictionSummary:
        """Run prediction and return results as PredictionSummary."""
        return self._service.predict(df, return_shap=return_shap)

    def save_model(self, path: str) -> str:
        """Save the trained model to the given path. Returns the path."""
        return self._service.save_model(path)

    def save_config(self, path: str) -> None:
        """Save current full config to YAML file."""
        import yaml  # type: ignore[import-untyped]

        full_config = self._service.build_config(dict(self.config))
        with open(path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False)

    def load_config(self, path: str) -> LizyWidget:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            loaded: dict[str, Any] = yaml.safe_load(f)

        self._apply_loaded_config(loaded)
        return self

    def _apply_loaded_config(self, loaded: dict[str, Any]) -> None:
        """Apply a parsed config dict to widget state (canonicalized via adapter)."""
        canonical = self._service.apply_loaded_config(loaded)
        self.df_info = self._service.get_df_info()
        self.config = canonical

    # ── Action dispatcher ─────────────────────────────────────

    @traitlets.observe("action")
    def _on_action(self, change: dict[str, Any]) -> None:
        action: dict[str, Any] = change["new"]
        if not action:
            return
        action_type: str = action.get("type", "")
        payload: dict[str, Any] = action.get("payload", {})

        handler = self._action_handlers.get(action_type)
        if handler is not None:
            handler(self, payload)

    def _handle_set_target(self, payload: dict[str, Any]) -> None:
        target = payload.get("target", "")
        if not target:
            return
        try:
            df_info = self._service.set_target(target)
            self.df_info = df_info
            self.status = "data_loaded"
            task = df_info.get("task")
            if task:
                self.config = self._service.apply_task_params(dict(self.config), task)
        except Exception as e:
            self.error = {"code": "TARGET_ERROR", "message": str(e)}

    def _handle_set_task(self, payload: dict[str, Any]) -> None:
        task = payload.get("task", "")
        if not task:
            return
        try:
            df_info = self._service.set_task(task)
            self.df_info = df_info
            if task:
                self.config = self._service.apply_task_params(dict(self.config), task)
        except Exception as e:
            self.error = {"code": "TASK_ERROR", "message": str(e)}

    def _handle_update_column(self, payload: dict[str, Any]) -> None:
        try:
            df_info = self._service.update_column(
                payload["name"],
                excluded=payload.get("excluded", False),
                col_type=payload.get("col_type", "numeric"),
            )
            self.df_info = df_info
        except Exception as e:
            self.error = {"code": "COLUMN_ERROR", "message": str(e)}

    def _handle_update_cv(self, payload: dict[str, Any]) -> None:
        try:
            df_info = self._service.update_cv(
                payload.get("strategy", "kfold"),
                payload.get("n_splits", 5),
                group_column=payload.get("group_column"),
                time_column=payload.get("time_column"),
                random_state=payload.get("random_state", 42),
                shuffle=payload.get("shuffle", True),
                gap=payload.get("gap", 0),
                purge_gap=payload.get("purge_gap", 0),
                embargo=payload.get("embargo", 0),
                train_size_max=payload.get("train_size_max"),
                test_size_max=payload.get("test_size_max"),
            )
            self.df_info = df_info
        except Exception as e:
            self.error = {"code": "CV_ERROR", "message": str(e)}

    def _handle_patch_config(self, payload: dict[str, Any]) -> None:
        raw_ops = payload.get("ops", [])
        if not raw_ops:
            return
        ops = [ConfigPatchOp(op=o["op"], path=o["path"], value=o.get("value")) for o in raw_ops]
        self.config = self._service.apply_config_patch(dict(self.config), ops)

    def _handle_fit(self, _payload: dict[str, Any]) -> None:
        self._run_job("fit")

    def _handle_tune(self, _payload: dict[str, Any]) -> None:
        self._run_job("tune")

    def _handle_cancel(self, _payload: dict[str, Any]) -> None:
        self._cancel_flag.set()

    def _handle_request_plot(self, payload: dict[str, Any]) -> None:
        plot_type = payload.get("plot_type", "")
        if not plot_type:
            return
        try:
            plot_data = self._service.get_plot(plot_type)
            self.send(
                {
                    "type": "plot_data",
                    "plot_type": plot_type,
                    "plotly_json": plot_data.plotly_json,
                }
            )
        except Exception as e:
            self.send(
                {
                    "type": "plot_error",
                    "plot_type": plot_type,
                    "message": str(e),
                }
            )

    def _handle_run_inference(self, payload: dict[str, Any]) -> None:
        if self._inference_df is None:
            self.error = {"code": "INFERENCE_ERROR", "message": "No inference data loaded"}
            return
        try:
            return_shap = payload.get("return_shap", False)
            result = self._service.predict(self._inference_df, return_shap=return_shap)
            records = result.predictions.to_dict(orient="records")
            self.inference_result = {
                "status": "completed",
                "rows": len(records),
                "data": records,
                "warnings": result.warnings,
            }
        except Exception as e:
            self.inference_result = {
                "status": "failed",
                "message": str(e),
            }

    def _handle_import_yaml(self, payload: dict[str, Any]) -> None:
        content = payload.get("content", "")
        if not content:
            return
        try:
            import yaml

            loaded: dict[str, Any] = yaml.safe_load(content)
            if not isinstance(loaded, dict):
                self.error = {"code": "IMPORT_ERROR", "message": "Invalid YAML content"}
                return
            self._apply_loaded_config(loaded)
        except Exception as e:
            self.error = {"code": "IMPORT_ERROR", "message": str(e)}

    def _handle_export_yaml(self, _payload: dict[str, Any]) -> None:
        try:
            import yaml

            full_config = self._service.build_config(dict(self.config))
            content = yaml.dump(full_config, default_flow_style=False)
            self.send({"type": "yaml_export", "content": content})
        except Exception as e:
            self.error = {"code": "EXPORT_ERROR", "message": str(e)}

    def _handle_raw_config(self, _payload: dict[str, Any]) -> None:
        try:
            import yaml

            if self._service.has_data() and self._service.has_target():
                full_config = self._service.build_config(dict(self.config))
            else:
                full_config = dict(self.config)
            content = yaml.dump(full_config, default_flow_style=False)
            self.send({"type": "raw_config", "content": content})
        except Exception as e:
            with contextlib.suppress(Exception):
                self.send({"type": "raw_config_error", "message": str(e)})
            self.error = {"code": "EXPORT_ERROR", "message": str(e)}

    def _handle_apply_best_params(self, payload: dict[str, Any]) -> None:
        params = payload.get("params", {})
        if not params:
            return

        # Restore tune-time config snapshot if available (P-005)
        if self._tune_config_snapshot is not None:
            current = copy.deepcopy(self._tune_config_snapshot)
            # Strip data/features/split/task — managed by df_info/service
            for key in ("data", "features", "split", "task"):
                current.pop(key, None)
        else:
            current = copy.deepcopy(dict(self.config))

        model_section = dict(current.get("model", {}))
        model_params = dict(model_section.get("params", {}))
        model_params.update(params)
        model_section["params"] = model_params
        current["model"] = model_section
        self.config = self._service.canonicalize_config(current)

    def _handle_request_inference_plot(self, payload: dict[str, Any]) -> None:
        plot_type = payload.get("plot_type", "")
        if not plot_type:
            return
        # Inference plots use prediction data, not fit model
        inference_data = self.inference_result.get("data", [])
        if not inference_data:
            self._handle_request_plot(payload)
            return
        try:
            predictions = pd.DataFrame(inference_data)
            plot_data = self._service.get_inference_plot(predictions, plot_type)
            self.send(
                {
                    "type": "plot_data",
                    "plot_type": plot_type,
                    "plotly_json": plot_data.plotly_json,
                }
            )
        except Exception:
            # Fall back to fit model plot if inference plot fails
            self._handle_request_plot(payload)

    _action_handlers: dict[str, Any] = {
        "set_target": _handle_set_target,
        "set_task": _handle_set_task,
        "update_column": _handle_update_column,
        "update_cv": _handle_update_cv,
        "patch_config": _handle_patch_config,
        "fit": _handle_fit,
        "tune": _handle_tune,
        "cancel": _handle_cancel,
        "request_plot": _handle_request_plot,
        "run_inference": _handle_run_inference,
        "apply_best_params": _handle_apply_best_params,
        "request_inference_plot": _handle_request_inference_plot,
        "import_yaml": _handle_import_yaml,
        "export_yaml": _handle_export_yaml,
        "raw_config": _handle_raw_config,
    }

    # ── Job execution ─────────────────────────────────────────

    def _run_job(self, job_type: str) -> None:
        if self.status == "running":
            return

        # Pre-execution data/target checks (BLUEPRINT §6.1)
        if not self._service.has_data():
            self.error = {"code": "NO_DATA", "message": "No data loaded. Call load(df) first."}
            self.status = "failed"
            return

        if not self._service.has_target():
            self.error = {
                "code": "NO_TARGET",
                "message": "No target selected. Call load(df, target=...) or set_target(col).",
            }
            self.status = "failed"
            return

        self._cancel_flag.clear()
        self._job_counter += 1
        self.job_type = job_type
        self.job_index = self._job_counter
        self.status = "running"
        self.progress = {"current": 0, "total": 0, "message": f"Starting {job_type}..."}
        self.elapsed_sec = 0.0
        self.error = {}

        full_config = self._service.prepare_run_config(dict(self.config), job_type=job_type)

        if job_type == "tune":
            self._tune_config_snapshot = copy.deepcopy(full_config)

        # Pre-execution validation (BLUEPRINT §9-3)
        errors = self._service.validate_config(full_config)
        if errors:
            self.error = {
                "code": "VALIDATION_ERROR",
                "message": errors[0]["message"],
                "details": errors,
            }
            self.status = "failed"
            return

        thread = threading.Thread(
            target=self._job_worker,
            args=(job_type, full_config),
            daemon=True,
        )
        self._job_thread = thread
        thread.start()

    def _job_worker(self, job_type: str, config: dict[str, Any]) -> None:
        start = time.monotonic()
        timer_stop = threading.Event()

        def tick_elapsed() -> None:
            while not timer_stop.is_set():
                self.elapsed_sec = round(time.monotonic() - start, 1)
                timer_stop.wait(1.0)

        timer = threading.Thread(target=tick_elapsed, daemon=True)
        timer.start()

        def on_progress(current: int, total: int, message: str) -> None:
            if self._cancel_flag.is_set():
                raise InterruptedError("Job cancelled by user")
            self.progress = {"current": current, "total": total, "message": message}
            self.elapsed_sec = round(time.monotonic() - start, 1)

        try:
            if job_type == "fit":
                summary = self._service.fit(config, on_progress=on_progress)
                normalized = self._normalize_metrics(self._service.get_evaluate_table())
                fold_details = self._service.get_split_summary()
                self.fit_summary = {
                    "metrics": normalized if normalized else summary.metrics,
                    "fold_count": summary.fold_count,
                    "fold_details": fold_details,
                    "params": summary.params,
                }
            elif job_type == "tune":
                n_trials = (
                    config.get("tuning", {}).get("optuna", {}).get("params", {}).get("n_trials", 50)
                )
                on_progress(0, n_trials, f"Tuning {n_trials} trials...")
                summary_t = self._service.tune(config, on_progress=on_progress)
                self.tune_summary = {
                    "best_params": summary_t.best_params,
                    "best_score": summary_t.best_score,
                    "trials": summary_t.trials,
                    "metric_name": summary_t.metric_name,
                    "direction": summary_t.direction,
                }
                # After tune, model MAY be fitted — guard evaluate/split calls (P-004 R3)
                try:
                    normalized = self._normalize_metrics(self._service.get_evaluate_table())
                    fold_details = self._service.get_split_summary()
                    if normalized:
                        self.fit_summary = {
                            "metrics": normalized,
                            "fold_count": len(fold_details),
                            "fold_details": fold_details,
                            "params": [],
                        }
                except Exception:
                    pass  # Tune-only: no fit results available

            self.available_plots = self._service.get_available_plots()
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.status = "completed"

        except InterruptedError:
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.error = {"code": "CANCELLED", "message": "Job cancelled by user"}
            self.status = "failed"

        except Exception as e:
            self.elapsed_sec = round(time.monotonic() - start, 1)
            # Distinguish adapter/backend errors from internal widget errors
            try:
                mod = getattr(type(e), "__module__", "") or ""
                code = "BACKEND_ERROR" if "lizyml" in mod.lower() else "INTERNAL_ERROR"
            except Exception:
                code = "INTERNAL_ERROR"
            self.error = {
                "code": code,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            self.status = "failed"

        finally:
            timer_stop.set()
            timer.join(timeout=2.0)

    # ── Config helpers ─────────────────────────────────────────

    @staticmethod
    def _normalize_metrics(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert evaluate_table records to {metric: {is, oos, oos_std}} for ScoreTable.

        evaluate_table() returns records with 'index' (metric name), 'if_mean', 'oof',
        and 'fold_0'...'fold_N-1' columns.
        """
        result: dict[str, Any] = {}
        if not eval_records:
            return result
        for record in eval_records:
            metric_name = str(record.get("index", record.get("metric", "")))
            if not metric_name:
                continue
            entry: dict[str, Any] = {}
            if "if_mean" in record:
                entry["is"] = record["if_mean"]
            if "oof" in record:
                entry["oos"] = record["oof"]
            # Compute OOS Std from per-fold columns
            fold_values = [
                v
                for k, v in record.items()
                if k.startswith("fold_") and isinstance(v, (int, float))
            ]
            if len(fold_values) > 1:
                entry["oos_std"] = statistics.stdev(fold_values)
            result[metric_name] = entry
        return result
