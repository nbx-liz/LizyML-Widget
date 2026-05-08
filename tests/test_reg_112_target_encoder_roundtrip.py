"""Regression test for #112 / P-030: lizyml 0.10 target_encoder round-trip.

Ensures the Adapter passes non-numeric classification labels through
``Model.fit`` -> ``Model.predict`` while preserving the original label
dtype, leveraging lizyml 0.10's ``FitResult.target_encoder``.

Two failure modes this guards against:

1. The Adapter could accidentally coerce ``result.pred`` to numeric int
   codes (loses dtype). Exercises the ``LizyMLAdapter.predict`` path
   end-to-end with a real lizyml backend.
2. ``task='regression'`` with a non-numeric target must surface the new
   ``TARGET_NOT_NUMERIC`` error code from lizyml as a Widget
   ``BACKEND_ERROR`` (no silent failure or generic LightGBM crash).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter


def _make_classification_df(*, seed: int = 42, n: int = 80) -> pd.DataFrame:
    """Toy multiclass dataset with non-numeric (object) labels."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "species": rng.choice(["Adelie", "Chinstrap", "Gentoo"], size=n),
        }
    )


@pytest.fixture()
def adapter() -> LizyMLAdapter:
    return LizyMLAdapter()


class TestNonNumericLabelRoundTrip:
    """LizyML 0.10 + non-numeric classification labels: predict preserves dtype."""

    def test_multiclass_predict_returns_original_label_dtype(self, adapter: LizyMLAdapter) -> None:
        df = _make_classification_df()
        config = adapter.initialize_config(task="multiclass")
        config["task"] = "multiclass"
        config["data"] = {"target": "species"}
        config["model"]["params"]["n_estimators"] = 5
        config["model"]["params"]["verbose"] = -1
        config["training"] = {"seed": 42}

        run_config = adapter.prepare_run_config(config, job_type="fit", task="multiclass")
        model = adapter.create_model(run_config, df)
        adapter.fit(model)

        # target_encoder is the lizyml 0.10 marker — must exist and carry classes_
        encoder = getattr(model.fit_result, "target_encoder", None)
        assert encoder is not None, "FitResult.target_encoder missing (lizyml<0.10?)"
        classes = list(getattr(encoder, "classes_", ()) or ())
        assert {"Adelie", "Chinstrap", "Gentoo"} <= set(classes)

        # Predict — pred column should hold the *original* string labels,
        # not the integer codes used internally.
        result = adapter.predict(model, df.drop(columns=["species"]))
        pred_col = result.predictions["pred"]
        sample = pred_col.iloc[:5].tolist()
        assert all(isinstance(v, str) for v in sample), (
            f"expected str-typed predictions, got {[type(v).__name__ for v in sample]}"
        )
        assert set(sample) <= {"Adelie", "Chinstrap", "Gentoo"}, (
            f"unexpected label outside training set: {sample}"
        )


class TestTargetNotNumericRaisesBackendError:
    """LizyML 0.10 raises TARGET_NOT_NUMERIC for regression on non-numeric y."""

    def test_regression_with_string_target_raises_backend_error(
        self, adapter: LizyMLAdapter
    ) -> None:
        from lizyml.core.exceptions import LizyMLError

        df = _make_classification_df()
        config = adapter.initialize_config(task="regression")
        config["task"] = "regression"
        config["data"] = {"target": "species"}
        config["model"]["params"]["n_estimators"] = 5
        config["model"]["params"]["verbose"] = -1
        config["training"] = {"seed": 42}

        run_config = adapter.prepare_run_config(config, job_type="fit", task="regression")
        model = adapter.create_model(run_config, df)
        with pytest.raises(LizyMLError) as excinfo:
            adapter.fit(model)

        # Code attribute carries the new error tag introduced in lizyml 0.10.
        code: Any = getattr(excinfo.value, "code", None)
        # Code may be an Enum (ErrorCode.TARGET_NOT_NUMERIC) or a bare string —
        # accept both shapes so this test survives lizyml-side refactors.
        code_str = getattr(code, "name", None) or str(code)
        assert "TARGET_NOT_NUMERIC" in code_str
