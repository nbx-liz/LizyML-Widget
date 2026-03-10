"""LizyWidget - anywidget-based notebook UI for LizyML."""

from __future__ import annotations

import importlib.resources
import json
import threading
import time
import traceback
from typing import Any

import anywidget
import pandas as pd
import traitlets

from .adapter import LizyMLAdapter
from .service import WidgetService


class LizyWidget(anywidget.AnyWidget):
    """LizyML notebook UI widget."""

    _esm = importlib.resources.files("lizyml_widget") / "static/widget.js"
    _css = importlib.resources.files("lizyml_widget") / "static/widget.css"

    # ── Python → JS traitlets ────────────────────────────────
    backend_info = traitlets.Dict({}).tag(sync=True)
    df_info = traitlets.Dict({}).tag(sync=True)
    config_schema = traitlets.Dict({}).tag(sync=True)
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._service = WidgetService(adapter=LizyMLAdapter())
        self._job_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._job_counter = 0
        self._inference_df: pd.DataFrame | None = None

        info = self._service.info
        self.backend_info = {"name": info.name, "version": info.version}

    # ── Public Python API ─────────────────────────────────────

    def load(self, df: pd.DataFrame, target: str | None = None) -> None:
        """Load a DataFrame into the widget."""
        df_info = self._service.load_data(df, target=target)
        self.df_info = df_info
        self.status = "data_loaded"
        self.error = {}
        self.fit_summary = {}
        self.tune_summary = {}
        self.available_plots = []

        schema = self._service.get_config_schema()
        self.config_schema = schema.json_schema

    def load_inference(self, df: pd.DataFrame) -> None:
        """Load a DataFrame for inference."""
        self._inference_df = df
        self.inference_result = {"status": "ready", "rows": len(df)}

    def set_config(self, config: dict[str, Any]) -> None:
        """Set config programmatically."""
        self.config = config

    def get_config(self) -> dict[str, Any]:
        """Get current config."""
        return dict(self.config)

    def get_fit_summary(self) -> dict[str, Any]:
        """Get last fit summary."""
        return dict(self.fit_summary)

    def get_tune_summary(self) -> dict[str, Any]:
        """Get last tune summary."""
        return dict(self.tune_summary)

    def get_model(self) -> Any:
        """Get the underlying trained model object."""
        return self._service.get_model()

    def predict(self, df: pd.DataFrame, *, return_shap: bool = False) -> pd.DataFrame:
        """Run prediction and return results as DataFrame."""
        result = self._service.predict(df, return_shap=return_shap)
        return result.predictions

    def save_config(self, path: str) -> None:
        """Save current full config to YAML file."""
        import yaml  # type: ignore[import-untyped]

        full_config = self._service.build_config(dict(self.config))
        with open(path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False)

    def load_config(self, path: str) -> None:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            loaded: dict[str, Any] = yaml.safe_load(f)

        data_section = loaded.pop("data", {})
        loaded.pop("features", {})  # consumed by service via build_config
        split_section = loaded.pop("split", {})

        # Apply data section to service
        if "target" in data_section and self._service._df is not None:
            self._service.set_target(data_section["target"])

        # Apply CV from split section
        if split_section:
            strategy = split_section.get("method", split_section.get("strategy", "kfold"))
            n_splits = split_section.get("n_splits", 5)
            group_col = split_section.get("group_col", split_section.get("group_column"))
            self._service.update_cv(strategy, n_splits, group_col)

        self.df_info = self._service._df_info

        # Remaining keys go to config
        self.config = loaded

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
        except Exception as e:
            self.error = {"code": "TARGET_ERROR", "message": str(e)}

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
                payload.get("group_column"),
            )
            self.df_info = df_info
        except Exception as e:
            self.error = {"code": "CV_ERROR", "message": str(e)}

    def _handle_update_config(self, payload: dict[str, Any]) -> None:
        self.config = payload.get("config", {})

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

    _action_handlers: dict[str, Any] = {
        "set_target": _handle_set_target,
        "update_column": _handle_update_column,
        "update_cv": _handle_update_cv,
        "update_config": _handle_update_config,
        "fit": _handle_fit,
        "tune": _handle_tune,
        "cancel": _handle_cancel,
        "request_plot": _handle_request_plot,
        "run_inference": _handle_run_inference,
    }

    # ── Job execution ─────────────────────────────────────────

    def _run_job(self, job_type: str) -> None:
        if self.status == "running":
            return

        self._cancel_flag.clear()
        self._job_counter += 1
        self.job_type = job_type
        self.job_index = self._job_counter
        self.status = "running"
        self.progress = {"current": 0, "total": 0, "message": f"Starting {job_type}..."}
        self.elapsed_sec = 0.0
        self.error = {}

        full_config = self._service.build_config(dict(self.config))

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

        try:
            if job_type == "fit":
                summary = self._service.fit(config)
                self.fit_summary = {
                    "metrics": summary.metrics,
                    "fold_count": summary.fold_count,
                    "params": summary.params,
                }
            elif job_type == "tune":
                summary_t = self._service.tune(config)
                self.tune_summary = {
                    "best_params": summary_t.best_params,
                    "best_score": summary_t.best_score,
                    "trials": summary_t.trials,
                    "metric_name": summary_t.metric_name,
                    "direction": summary_t.direction,
                }

            self.available_plots = self._service.get_available_plots()
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.status = "completed"

        except Exception as e:
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.error = {
                "code": "BACKEND_ERROR",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            self.status = "failed"

        finally:
            timer_stop.set()
            timer.join(timeout=2.0)

    # ── Serialization helpers ─────────────────────────────────

    def _serialize_fit_summary(self, summary: Any) -> str:
        return json.dumps(
            {
                "metrics": summary.metrics,
                "fold_count": summary.fold_count,
                "params": summary.params,
            }
        )
