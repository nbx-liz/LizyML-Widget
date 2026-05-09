"""Regression test for #112 / P-030: lizyml 0.11 smape / wape regression metrics.

Two seams to lock down so users can pick the new metrics from the Widget UI:

1. The ``model_metric.regression`` option set in ``BackendContract.ui_schema``
   exposes ``smape`` and ``wape`` so they appear as metric chips in
   Search Space / Model tab.
2. ``MODEL_METRIC_TO_EVAL`` carries identity mappings for both names so tune
   direction resolution (``minimize``) works when ``model.params.metric``
   is set to ``smape`` or ``wape``.

These cover the bookkeeping side. End-to-end propagation through
``Model.fit(metric=['smape', 'wape'])`` -> ``eval_history`` is already
covered by lizyml's own test suite; we only assert the Widget passes the
metric name through to the LightGBM bridge without mangling it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.adapter_params import MODEL_METRIC_TO_EVAL


@pytest.fixture()
def adapter() -> LizyMLAdapter:
    return LizyMLAdapter()


class TestModelMetricOptionSet:
    """smape / wape must be selectable as model.params.metric for regression."""

    def test_smape_in_regression_model_metric(self, adapter: LizyMLAdapter) -> None:
        contract = adapter.get_backend_contract()
        regression = contract.ui_schema["option_sets"]["model_metric"]["regression"]
        assert "smape" in regression

    def test_wape_in_regression_model_metric(self, adapter: LizyMLAdapter) -> None:
        contract = adapter.get_backend_contract()
        regression = contract.ui_schema["option_sets"]["model_metric"]["regression"]
        assert "wape" in regression


class TestEvalMetricMapping:
    """smape / wape need identity mappings so tune direction resolves correctly."""

    def test_smape_identity_mapping(self) -> None:
        assert MODEL_METRIC_TO_EVAL.get("smape") == "smape"

    def test_wape_identity_mapping(self) -> None:
        assert MODEL_METRIC_TO_EVAL.get("wape") == "wape"


class TestRegressionFitWithSmapeWape:
    """End-to-end: fit a regression model with metric=['smape', 'wape']."""

    def _make_regression_df(self, *, seed: int = 42, n: int = 80) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 2.0 * x1 + 0.5 * x2 + rng.normal(scale=0.1, size=n)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    def test_smape_wape_appear_in_eval_history(self, adapter: LizyMLAdapter) -> None:
        df = self._make_regression_df()
        config = adapter.initialize_config(task="regression")
        config["task"] = "regression"
        config["data"] = {"target": "y"}
        config["model"]["params"]["n_estimators"] = 5
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["smape", "wape"]
        config["training"] = {"seed": 42}

        run_config = adapter.prepare_run_config(config, job_type="fit", task="regression")
        model = adapter.create_model(run_config, df)
        adapter.fit(model)

        history = model.fit_result.history[0].get("eval_history", {})
        all_metrics: set[str] = set()
        for ds_metrics in history.values():
            all_metrics.update(ds_metrics.keys())
        assert "smape" in all_metrics, f"smape missing from eval_history; got {all_metrics}"
        assert "wape" in all_metrics, f"wape missing from eval_history; got {all_metrics}"
