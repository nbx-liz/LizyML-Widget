"""Tests for tune-state IPC on LizyMLAdapter (P-037, #152).

Pins the contract introduced for restoring ``model._tuning_result`` (and
best-effort ``_study``) across the subprocess boundary so the parent
process can render the optimization-history plot without re-running the
fit.

INV-2: ``restore_tune_state`` must keep ``is_model_fitted(model) == False``.
INV-3: ``restore_tune_state`` must set ``model._tuning_result`` to a
       non-None value so ``available_plots`` includes
       ``"optimization-history"``.
INV-4: ``export_tune_state`` is well-defined for tune-only models and
       raises clearly when the model has no tune state.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.adapter_results import is_model_fitted, list_available_plots


def _fake_tuning_result() -> Any:
    """Build a TuningResult-like dataclass instance that is fully pickleable."""
    from lizyml.core.types.tuning_result import TrialResult, TuningResult

    return TuningResult(
        best_model_params={"learning_rate": 0.05},
        best_smart_params={},
        best_training_params={"num_iterations": 100},
        best_score=0.987,
        trials=[
            TrialResult(number=0, params={"learning_rate": 0.05}, score=0.987, state="complete"),
            TrialResult(number=1, params={"learning_rate": 0.10}, score=0.950, state="complete"),
        ],
        metric_name="auc",
        direction="maximize",
    )


def _attach_tune_state_defaults(model: Any) -> None:
    """Set picklable values for P-038 round-bookkeeping attrs on a mock model.

    ``MagicMock`` auto-creates attributes as nested mocks, which break
    ``pickle.dumps`` inside ``export_tune_state``. The P-037-era tests in
    this module never set these attributes; this helper makes them
    deterministic and pickle-safe so the same fixtures keep working
    after P-038 added ``_rounds`` / ``_round_number`` / ``_space`` /
    ``_used_default_space`` to the export blob.
    """
    model._round_number = 1
    model._rounds = []
    model._space = None
    model._used_default_space = False


class TestExportTuneState:
    """``export_tune_state`` writes a pickle blob containing ``_tuning_result``."""

    def test_writes_blob_with_tuning_result(self, tmp_path: Path) -> None:
        adapter = LizyMLAdapter()
        model = MagicMock()
        model._tuning_result = _fake_tuning_result()
        model._study = None  # InMemory absent
        _attach_tune_state_defaults(model)

        out_path = tmp_path / "tune_state.pkl"
        adapter.export_tune_state(model, str(out_path))

        assert out_path.exists()
        with out_path.open("rb") as f:
            blob = pickle.load(f)  # noqa: S301 — trusted, written by us
        assert "tuning_result" in blob
        assert blob["tuning_result"].best_score == pytest.approx(0.987)

    def test_raises_when_no_tune_state(self, tmp_path: Path) -> None:
        """INV-4 corollary: a model that never tuned has no exportable state."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model._tuning_result = None
        model._study = None

        with pytest.raises(ValueError, match="no tune state"):
            adapter.export_tune_state(model, str(tmp_path / "x.pkl"))

    def test_bundles_pickleable_study_best_effort(self, tmp_path: Path) -> None:
        """``_study`` is bundled when pickleable, omitted otherwise (warn)."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model._tuning_result = _fake_tuning_result()
        # InMemoryStorage is pickleable; use a simple sentinel-pickleable object.
        model._study = {"trials": [1, 2, 3]}  # dict pickles cleanly
        _attach_tune_state_defaults(model)

        out = tmp_path / "tune_state.pkl"
        adapter.export_tune_state(model, str(out))

        with out.open("rb") as f:
            blob = pickle.load(f)  # noqa: S301
        assert blob.get("study") == {"trials": [1, 2, 3]}

    def test_skips_unpickleable_study(self, tmp_path: Path) -> None:
        """If pickling _study raises, blob is written without study (warn-only)."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model._tuning_result = _fake_tuning_result()

        # Lambdas are not pickleable; attribute read returns one.
        class _UnpickleableStudy:
            def __getstate__(self) -> Any:
                msg = "study cannot be pickled"
                raise TypeError(msg)

        model._study = _UnpickleableStudy()
        _attach_tune_state_defaults(model)

        out = tmp_path / "tune_state.pkl"
        adapter.export_tune_state(model, str(out))

        with out.open("rb") as f:
            blob = pickle.load(f)  # noqa: S301
        assert blob.get("study") is None
        assert "tuning_result" in blob


class TestRestoreTuneState:
    """``restore_tune_state`` sets ``_tuning_result`` (and ``_study`` if present)
    on a freshly-created model, satisfying INV-2 and INV-3."""

    def test_restore_sets_tuning_result(self, tmp_path: Path) -> None:
        adapter = LizyMLAdapter()

        # Write a blob the way export would
        blob = {"tuning_result": _fake_tuning_result(), "study": None}
        path = tmp_path / "blob.pkl"
        with path.open("wb") as f:
            pickle.dump(blob, f)

        # Fresh model: no tune state, no fit state
        model = MagicMock()
        model._tuning_result = None
        model._study = None
        model.fit_result = None

        adapter.restore_tune_state(model, str(path))

        # INV-3: tuning result is set
        assert model._tuning_result is not None
        assert model._tuning_result.best_score == pytest.approx(0.987)

    def test_restore_keeps_model_unfitted(self, tmp_path: Path) -> None:
        """INV-2: restored model must NOT report as fitted."""
        adapter = LizyMLAdapter()

        blob = {"tuning_result": _fake_tuning_result(), "study": None}
        path = tmp_path / "blob.pkl"
        with path.open("wb") as f:
            pickle.dump(blob, f)

        model = MagicMock()
        model._tuning_result = None
        model._study = None
        # Crucially: simulate "no fit" by making fit_result None.
        model.fit_result = None

        adapter.restore_tune_state(model, str(path))

        assert is_model_fitted(model) is False

    def test_restore_then_available_plots_includes_optimization_history(
        self, tmp_path: Path
    ) -> None:
        """INV-3 corollary: after restore, ``optimization-history`` becomes
        available even though the model is still unfit."""
        adapter = LizyMLAdapter()

        blob = {"tuning_result": _fake_tuning_result(), "study": None}
        path = tmp_path / "blob.pkl"
        with path.open("wb") as f:
            pickle.dump(blob, f)

        model = MagicMock()
        model._tuning_result = None
        model._study = None
        model.fit_result = None

        adapter.restore_tune_state(model, str(path))

        plots = list_available_plots(model, task="binary")
        assert "optimization-history" in plots
        # Sanity: fit-dependent plots are still excluded.
        assert "learning-curve" not in plots

    def test_restore_with_study(self, tmp_path: Path) -> None:
        """When the blob carries a study, it is also restored (#128 prep)."""
        adapter = LizyMLAdapter()

        blob = {"tuning_result": _fake_tuning_result(), "study": {"trials": [1, 2]}}
        path = tmp_path / "blob.pkl"
        with path.open("wb") as f:
            pickle.dump(blob, f)

        model = MagicMock()
        model._tuning_result = None
        model._study = None
        model.fit_result = None

        adapter.restore_tune_state(model, str(path))

        assert model._study == {"trials": [1, 2]}


class TestRoundTrip:
    """Export → Restore round-trip preserves tuning result equality."""

    def test_round_trip_preserves_tuning_result(self, tmp_path: Path) -> None:
        adapter = LizyMLAdapter()

        original = _fake_tuning_result()
        producer = MagicMock()
        producer._tuning_result = original
        producer._study = None
        _attach_tune_state_defaults(producer)

        path = tmp_path / "rt.pkl"
        adapter.export_tune_state(producer, str(path))

        consumer = MagicMock()
        consumer._tuning_result = None
        consumer._study = None
        consumer.fit_result = None

        adapter.restore_tune_state(consumer, str(path))

        assert consumer._tuning_result == original
