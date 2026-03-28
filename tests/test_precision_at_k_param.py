"""Tests for precision_at_k k parameter handling (P-026 extension).

Verifies:
1. prepare_run_config converts _precision_at_k_k to MetricEntry dict
2. strip_for_backend removes _precision_at_k_k from model.params
3. When precision_at_k is in metric without _precision_at_k_k, default k=10
4. When precision_at_k is NOT in metric, _precision_at_k_k is ignored
5. E2E: fit with custom k produces correct Learning Curve history key
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from lizyml_widget.adapter import LizyMLAdapter


@pytest.fixture(scope="module")
def binary_df() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame


class TestPrepareRunConfigMetricEntry:
    """prepare_run_config should convert _precision_at_k_k to MetricEntry."""

    def test_precision_at_k_with_custom_k(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "params": {
                    "metric": ["auc", "precision_at_k"],
                    "_precision_at_k_k": 20,
                },
            },
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        metric = result["model"]["params"]["metric"]

        # precision_at_k should be converted to MetricEntry dict
        assert "precision_at_k" not in metric, (
            "plain 'precision_at_k' should be replaced by MetricEntry dict"
        )
        entry = next(
            (m for m in metric if isinstance(m, dict) and "precision_at_k" in m),
            None,
        )
        assert entry is not None, f"MetricEntry dict not found in metric: {metric}"
        assert entry["precision_at_k"]["k"] == 20

        # auc should remain as plain string
        assert "auc" in metric

    def test_precision_at_k_default_k(self) -> None:
        """Without _precision_at_k_k, precision_at_k stays as plain string."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "params": {"metric": ["auc", "precision_at_k"]},
            },
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        metric = result["model"]["params"]["metric"]

        # Without _precision_at_k_k, plain string remains (default k=10)
        assert "precision_at_k" in metric

    def test_precision_at_k_k_without_metric_is_ignored(self) -> None:
        """_precision_at_k_k is stripped even if precision_at_k not in metric."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "params": {
                    "metric": ["auc"],
                    "_precision_at_k_k": 20,
                },
            },
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        params = result["model"]["params"]

        # _precision_at_k_k should be stripped
        assert "_precision_at_k_k" not in params
        # metric should be unchanged
        assert params["metric"] == ["auc"]


class TestPrecisionAtKStripped:
    """_precision_at_k_k must not reach the backend."""

    def test_prepare_run_config_strips_precision_at_k_k(self) -> None:
        """_precision_at_k_k is removed by _convert_metric_entries before strip_for_backend."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "params": {
                    "metric": ["auc"],
                    "_precision_at_k_k": 20,
                },
            },
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        assert "_precision_at_k_k" not in result.get("model", {}).get("params", {})


class TestAdapterPlotDisplayName:
    """adapter.plot() should convert plain metric name to display name for feval metrics."""

    def test_plot_filter_with_feval_display_name(self, binary_df: pd.DataFrame) -> None:
        """precision_at_k in plot filter should match history key 'precision_at_k (k=20)'."""
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 5, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 10
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "precision_at_k"]
        config["model"]["params"]["_precision_at_k_k"] = 20

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, binary_df)
        adapter.fit(model)

        import json

        # Filter by "precision_at_k (k=20)" display name should work
        result = adapter.plot(model, "learning-curve", metrics=["precision_at_k (k=20)"])
        spec = json.loads(result.plotly_json)
        annotations = spec.get("layout", {}).get("annotations", [])
        titles = [a.get("text", "") for a in annotations]
        assert any("precision_at_k" in t for t in titles), (
            f"Expected precision_at_k in titles: {titles}"
        )


class TestModelMetricsExtraction:
    """fitSummary.params should provide both native and feval metric names."""

    def test_fit_summary_contains_feval_metrics(self, binary_df: pd.DataFrame) -> None:
        """fit_summary.params should include feval_metrics row."""
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 5, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 10
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "precision_at_k"]
        config["model"]["params"]["_precision_at_k_k"] = 20

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, binary_df)
        summary = adapter.fit(model)

        # params should have both "metric" and "feval_metrics" rows
        metric_row = next((r for r in summary.params if r.get("parameter") == "metric"), None)
        feval_row = next((r for r in summary.params if r.get("parameter") == "feval_metrics"), None)

        assert metric_row is not None
        assert feval_row is not None
        assert "precision_at_k" in feval_row["value"]


class TestPrecisionAtKE2E:
    """E2E: fit with custom k records correct key in Learning Curve."""

    def test_fit_with_custom_k(self, binary_df: pd.DataFrame) -> None:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 5, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 10
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "precision_at_k"]
        config["model"]["params"]["_precision_at_k_k"] = 20

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")

        # Verify MetricEntry conversion happened
        metric = run_config["model"]["params"]["metric"]
        entry = next(
            (m for m in metric if isinstance(m, dict) and "precision_at_k" in m),
            None,
        )
        assert entry is not None
        assert entry["precision_at_k"]["k"] == 20

        # Fit and check history
        model = adapter.create_model(run_config, binary_df)
        adapter.fit(model)

        history = model.fit_result.history[0].get("eval_history", {})
        all_keys: set[str] = set()
        for ds_metrics in history.values():
            all_keys.update(ds_metrics.keys())

        assert "auc" in all_keys
        # precision_at_k with k=20 should appear as "precision_at_k (k=20)"
        pak_key = next((k for k in all_keys if "precision_at_k" in k), None)
        assert pak_key is not None, f"precision_at_k not in history: {all_keys}"
        assert "(k=20)" in pak_key, f"Expected k=20 in key, got: {pak_key}"
