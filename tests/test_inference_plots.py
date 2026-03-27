"""Tests for inference plot generation (prediction-distribution, shap-summary)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter


@pytest.fixture()
def adapter() -> LizyMLAdapter:
    return LizyMLAdapter()


class TestPredictionDistribution:
    def test_prediction_distribution_returns_plotly_json(self, adapter: LizyMLAdapter) -> None:
        predictions = pd.DataFrame({"pred": [0.1, 0.5, 0.9, 0.3, 0.7]})
        result = adapter.plot_inference(predictions, "prediction-distribution")
        assert result.plotly_json is not None
        parsed = json.loads(result.plotly_json)
        assert "data" in parsed
        assert "layout" in parsed

    def test_prediction_distribution_handles_custom_column_name(
        self, adapter: LizyMLAdapter
    ) -> None:
        # The adapter looks for columns starting with "pred"
        predictions = pd.DataFrame({"prediction_score": [0.1, 0.5, 0.9]})
        result = adapter.plot_inference(predictions, "prediction-distribution")
        assert result.plotly_json is not None
        parsed = json.loads(result.plotly_json)
        assert "data" in parsed


class TestShapSummary:
    def test_shap_summary_returns_plotly_json(self, adapter: LizyMLAdapter) -> None:
        predictions = pd.DataFrame(
            {
                "pred": [0.1, 0.5, 0.9],
                "shap_feature_a": [0.3, -0.1, 0.2],
                "shap_feature_b": [-0.5, 0.4, 0.1],
            }
        )
        result = adapter.plot_inference(predictions, "shap-summary")
        assert result.plotly_json is not None
        parsed = json.loads(result.plotly_json)
        assert "data" in parsed
        assert "layout" in parsed
        # Verify feature names are cleaned (shap_ prefix removed)
        bar_data = parsed["data"][0]
        assert "feature_a" in bar_data["y"] or "feature_b" in bar_data["y"]

    def test_shap_summary_raises_without_shap_columns(self, adapter: LizyMLAdapter) -> None:
        predictions = pd.DataFrame({"pred": [0.1, 0.5, 0.9], "other_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="No SHAP"):
            adapter.plot_inference(predictions, "shap-summary")


class TestUnknownPlotType:
    def test_unknown_inference_plot_type_raises(self, adapter: LizyMLAdapter) -> None:
        predictions = pd.DataFrame({"pred": [0.1]})
        with pytest.raises(ValueError, match="Unknown inference plot type"):
            adapter.plot_inference(predictions, "nonexistent-plot")
