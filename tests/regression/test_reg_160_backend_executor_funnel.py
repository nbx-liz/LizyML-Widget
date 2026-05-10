"""Funnel-contract tests for P-039 Phase 3 (BackendExecutor).

The architectural guarantee for Phase 3 is that **every caller-thread
ML library call site** in the widget routes through the executor.
Phase 4 (lint rule) will enforce this against future changes; this
file pins the current state so a regression that bypasses the
executor surfaces in fast tier instead of as silent INV-G drift.

The contracts:

- ``service.predict`` calls ``executor.run_ml(..., ml_kind="predict")``
- ``service.get_plot("feature-importance-shap")`` calls ``run_ml(..., ml_kind="plot_shap")``
- ``service.get_plot("<other>")`` calls ``run_ml(..., ml_kind="plot_other")``
- ``service.get_inference_plot(..., "shap-summary")`` calls ``run_ml(..., ml_kind="plot_shap")``
- ``service.get_inference_plot(..., "<other>")`` calls ``run_ml(..., ml_kind="plot_other")``

We patch the executor's ``run_ml`` to assert the funnel without
running the underlying mock adapter twice. The end-state INV-G
transitions are pinned separately by ``test_reg_160_inv_g_runtime_guard``;
this file is the funnel-shape contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_service_with_model() -> Any:
    from lizyml_widget.service import WidgetService

    adapter = MagicMock()
    adapter.predict.return_value = MagicMock()
    adapter.plot.return_value = MagicMock()
    adapter.plot_inference.return_value = MagicMock()
    svc = WidgetService(adapter=adapter)
    svc._model = MagicMock()
    return svc, adapter


class TestPredictFunnel:
    def test_predict_routes_through_executor_with_predict_kind(self) -> None:
        svc, adapter = _make_service_with_model()
        captured: dict[str, Any] = {}

        original = svc._executor.run_ml

        def spy(op: Any, *, ml_kind: str) -> Any:
            captured["ml_kind"] = ml_kind
            return original(op, ml_kind=ml_kind)  # type: ignore[arg-type]

        with patch.object(svc._executor, "run_ml", side_effect=spy) as mock_run:
            svc.predict(pd.DataFrame({"f1": [1, 2]}))

        assert mock_run.call_count == 1
        assert captured["ml_kind"] == "predict"
        adapter.predict.assert_called_once()

    def test_predict_does_not_call_adapter_outside_executor(self) -> None:
        """If ``run_ml`` is intercepted to *not* call the operation,
        the adapter must not be touched. This catches a refactor that
        accidentally calls the adapter twice (once in the funnel and
        once outside)."""
        svc, adapter = _make_service_with_model()

        with patch.object(svc._executor, "run_ml", return_value=MagicMock()) as mock_run:
            svc.predict(pd.DataFrame({"f1": [1, 2]}))

        mock_run.assert_called_once()
        adapter.predict.assert_not_called()


class TestGetPlotFunnel:
    @pytest.mark.parametrize(
        "plot_type, expected_kind",
        [
            ("feature-importance-shap", "plot_shap"),
            ("learning-curve", "plot_other"),
            ("optimization-history", "plot_other"),
        ],
    )
    def test_get_plot_routes_through_executor_with_correct_kind(
        self, plot_type: str, expected_kind: str
    ) -> None:
        svc, adapter = _make_service_with_model()
        captured: dict[str, Any] = {}

        original = svc._executor.run_ml

        def spy(op: Any, *, ml_kind: str) -> Any:
            captured["ml_kind"] = ml_kind
            return original(op, ml_kind=ml_kind)  # type: ignore[arg-type]

        with patch.object(svc._executor, "run_ml", side_effect=spy) as mock_run:
            svc.get_plot(plot_type)

        assert mock_run.call_count == 1
        assert captured["ml_kind"] == expected_kind
        adapter.plot.assert_called_once()


class TestGetInferencePlotFunnel:
    @pytest.mark.parametrize(
        "plot_type, expected_kind",
        [
            ("shap-summary", "plot_shap"),
            ("prediction-distribution", "plot_other"),
        ],
    )
    def test_get_inference_plot_routes_through_executor_with_correct_kind(
        self, plot_type: str, expected_kind: str
    ) -> None:
        svc, adapter = _make_service_with_model()
        captured: dict[str, Any] = {}

        original = svc._executor.run_ml

        def spy(op: Any, *, ml_kind: str) -> Any:
            captured["ml_kind"] = ml_kind
            return original(op, ml_kind=ml_kind)  # type: ignore[arg-type]

        with patch.object(svc._executor, "run_ml", side_effect=spy) as mock_run:
            svc.get_inference_plot(pd.DataFrame({"pred": [0.1]}), plot_type)

        assert mock_run.call_count == 1
        assert captured["ml_kind"] == expected_kind
        adapter.plot_inference.assert_called_once()

    def test_inference_plot_attribute_error_translates_to_type_error(self) -> None:
        """Existing AttributeError → TypeError translation must survive
        the executor refactor (regression for adapter without
        ``plot_inference``)."""
        svc, adapter = _make_service_with_model()
        adapter.plot_inference.side_effect = AttributeError("no plot_inference on this adapter")

        with pytest.raises(TypeError, match="Inference plots not supported by this adapter"):
            svc.get_inference_plot(pd.DataFrame({"pred": [0.1]}), "shap-summary")


class TestExecutorIsServiceOwned:
    """``service._executor`` is the canonical executor instance.

    Phase 4 lint will rely on this — every ML call must reach the
    executor stored on the service that constructed the adapter calls.
    A future refactor must not introduce a per-call executor or a
    module-global instance.
    """

    def test_service_holds_executor_reference(self) -> None:
        svc, _ = _make_service_with_model()
        from lizyml_widget.backend_executor import BackendExecutor

        assert isinstance(svc._executor, BackendExecutor)

    def test_executor_back_reference_points_to_owning_service(self) -> None:
        svc, _ = _make_service_with_model()
        assert svc._executor._service is svc
