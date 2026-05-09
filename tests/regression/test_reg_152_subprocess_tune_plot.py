"""Regression test for issue #152 / P-037.

Subprocess tune (the new default after P-036 / #147) must persist enough
tune state across the IPC boundary that the parent process can render
``optimization-history`` without re-fitting. This test fails on
commit ``5497eab`` (PR #151 tip) and passes after P-037 lands.

Strategy: drive the public ``SubprocessJobRunner`` end-to-end with
``run_job_subprocess`` mocked so we don't actually start a child Python
process, but every layer above the IPC boundary (job_runner ->
service.restore_tune_state_from_path -> adapter.restore_tune_state ->
service.get_plot) is exercised in-process and asserts the bug is fixed.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.job_runner import JobSpec, SubprocessJobRunner
from lizyml_widget.service import WidgetService
from lizyml_widget.subprocess_runner import SubprocessJobResult


def _fake_tuning_result() -> Any:
    from lizyml.core.types.tuning_result import TrialResult, TuningResult

    return TuningResult(
        best_model_params={"learning_rate": 0.05},
        best_smart_params={},
        best_training_params={"num_iterations": 100},
        best_score=0.987,
        trials=[
            TrialResult(number=0, params={"learning_rate": 0.05}, score=0.987, state="complete"),
        ],
        metric_name="auc",
        direction="maximize",
    )


def _make_service_with_data() -> tuple[WidgetService, dict[str, Any]]:
    svc = WidgetService(adapter=LizyMLAdapter())
    # Multiple feature columns so build_config does not strip everything.
    df = pd.DataFrame(
        {
            "f1": list(range(20)),
            "f2": [i * 0.5 for i in range(20)],
            "f3": [i % 3 for i in range(20)],
            "y": [0, 1] * 10,
        }
    )
    svc.load_data(df, target="y")
    # Build the same canonical config the widget would hand to the runner.
    run_config = svc.prepare_run_config({}, job_type="tune")
    return svc, run_config


def _fake_subprocess_writes_blob(target_path: str) -> None:
    """Mimic the real subprocess: write a tune-state blob at the path the
    parent provided via ``tune_state_out_path``."""
    blob = {"tuning_result": _fake_tuning_result(), "study": None}
    with open(target_path, "wb") as f:  # noqa: PTH123
        pickle.dump(blob, f)


def test_subprocess_tune_renders_optimization_history_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: subprocess tune → parent renders ``optimization-history``.

    Pre-fix (commit 5497eab): subprocess tune returns ``model_path=None`` because
    ``model.export()`` raises MODEL_NOT_FIT for tune-only state, the exception
    is silently swallowed in ``_subprocess_entry``, and the parent's
    ``service._model`` stays None. ``service.get_plot("optimization-history")``
    therefore raises ``ValueError("No trained model")``.

    Post-fix: ``run_job_subprocess`` returns a ``tune_state_path``;
    ``SubprocessJobRunner`` calls ``service.restore_tune_state_from_path``,
    which uses ``adapter.create_model`` + ``adapter.restore_tune_state`` to
    inject ``_tuning_result`` onto a freshly created model. The parent then
    serves the plot via the adapter's existing ``model.tuning_plot()`` path.
    """
    svc, run_config = _make_service_with_data()

    # Mock the subprocess call. Mirror the real contract: the entry point
    # writes the blob at the parent-provided ``tune_state_out_path`` and
    # echoes that same path back in the result.
    def fake_run_job_subprocess(**kwargs: Any) -> SubprocessJobResult:
        out_path = kwargs["tune_state_out_path"]
        _fake_subprocess_writes_blob(out_path)
        return SubprocessJobResult(
            job_type="tune",
            fit_summary={},
            tune_summary={
                "best_params": {"learning_rate": 0.05},
                "best_score": 0.987,
                "trials": [
                    {"number": 0, "params": {}, "score": 0.987, "state": "complete"},
                ],
                "metric_name": "auc",
                "direction": "maximize",
            },
            eval_table=[],
            split_summary=[],
            available_plots=["optimization-history"],
            model_path=None,
            tune_state_path=out_path,
        )

    monkeypatch.setattr(
        "lizyml_widget.job_runner.run_job_subprocess",
        fake_run_job_subprocess,
    )

    runner = SubprocessJobRunner(svc)
    spec = JobSpec(
        job_type="tune",
        config=run_config,
        retune_kwargs=None,
        ui_snapshot={},
    )
    result = runner.run(spec, on_progress=lambda *a, **kw: None, cancel_event=threading.Event())

    assert result.job_type == "tune"
    assert "optimization-history" in result.available_plots

    # The parent must now own a tune-restored model that can render the plot.
    plot = svc.get_plot("optimization-history")
    assert plot.plotly_json  # non-empty string


def test_subprocess_tune_state_path_is_cleaned_up_after_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-5: tune state file is removed after the parent finishes loading it.

    The runner is responsible for unlinking the path in the ``finally``
    branch, mirroring how ``model_path`` is handled.
    """
    svc, run_config = _make_service_with_data()
    captured_path: dict[str, str] = {}

    def fake_run_job_subprocess(**kwargs: Any) -> SubprocessJobResult:
        out_path = kwargs["tune_state_out_path"]
        captured_path["path"] = out_path
        _fake_subprocess_writes_blob(out_path)
        return SubprocessJobResult(
            job_type="tune",
            fit_summary={},
            tune_summary={
                "best_params": {},
                "best_score": 0.0,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
            },
            eval_table=[],
            split_summary=[],
            available_plots=["optimization-history"],
            model_path=None,
            tune_state_path=out_path,
        )

    monkeypatch.setattr(
        "lizyml_widget.job_runner.run_job_subprocess",
        fake_run_job_subprocess,
    )

    runner = SubprocessJobRunner(svc)
    spec = JobSpec(job_type="tune", config=run_config, ui_snapshot={})
    runner.run(spec, on_progress=lambda *a, **kw: None, cancel_event=threading.Event())

    assert "path" in captured_path, "subprocess mock did not receive a path"
    assert not Path(captured_path["path"]).exists(), "tune state file must be removed after run"


def test_subprocess_tune_handles_missing_tune_state_path_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-1 (missing path): when subprocess returns ``tune_state_path=None``
    (e.g., a pre-P-037 child binary or export failure), the runner does NOT
    crash; the tune summary is still recorded so ``apply_best_params`` works.
    """
    svc, run_config = _make_service_with_data()

    def fake_run_job_subprocess(**_kwargs: Any) -> SubprocessJobResult:
        return SubprocessJobResult(
            job_type="tune",
            fit_summary={},
            tune_summary={
                "best_params": {"learning_rate": 0.05},
                "best_score": 0.5,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
            },
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
            tune_state_path=None,
        )

    monkeypatch.setattr(
        "lizyml_widget.job_runner.run_job_subprocess",
        fake_run_job_subprocess,
    )

    runner = SubprocessJobRunner(svc)
    spec = JobSpec(job_type="tune", config=run_config, ui_snapshot={})
    result = runner.run(spec, on_progress=lambda *a, **kw: None, cancel_event=threading.Event())

    assert result.tune_summary["best_score"] == pytest.approx(0.5)
    # Record was still written so apply_best_params works without the plot.
    assert svc._last_tune_summary is not None  # noqa: SLF001


def test_subprocess_tune_handles_corrupt_tune_state_blob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """INV-1 (corrupt path): if the blob fails to load, the runner logs a
    warning and continues with the tune summary recorded. The widget remains
    operable; only the plot is unavailable."""
    svc, run_config = _make_service_with_data()

    bad_path = tmp_path / "broken.pkl"
    bad_path.write_bytes(b"this is not a pickle stream")

    def fake_run_job_subprocess(**_kwargs: Any) -> SubprocessJobResult:
        return SubprocessJobResult(
            job_type="tune",
            fit_summary={},
            tune_summary={
                "best_params": {},
                "best_score": 0.0,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
            },
            eval_table=[],
            split_summary=[],
            available_plots=["optimization-history"],
            model_path=None,
            tune_state_path=str(bad_path),
        )

    monkeypatch.setattr(
        "lizyml_widget.job_runner.run_job_subprocess",
        fake_run_job_subprocess,
    )

    runner = SubprocessJobRunner(svc)
    spec = JobSpec(job_type="tune", config=run_config, ui_snapshot={})
    runner.run(spec, on_progress=lambda *a, **kw: None, cancel_event=threading.Event())

    # No exception reaches the widget; tune summary still recorded.
    assert svc._last_tune_summary is not None  # noqa: SLF001
