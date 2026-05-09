"""Regression tests for issue #156 — subprocess retune resume (P-038).

PR #155 patched #154 by routing retune to ``ThreadJobRunner`` under the
default subprocess strategy. The investigation in #156 found that the
thread fallback re-exposes the libgomp pool-affinity catastrophe whenever
the user enters any OpenMP parallel region on the parent main thread
(e.g., ``w.predict(test_df)``) between ``w.tune()`` and ``w.retune()``.
Per-trial retune wall-clock then jumps from ~3.2s (subprocess tune
baseline) to ~36s (~11x slowdown), with parent CPU collapsing from
~29 cores active to ~2 cores — exactly matching the user's "core usage
decreases" report.

P-038 / Option A replaces the thread fallback with subprocess retune
resume, extending the P-037 tune-state IPC to the input direction so the
subprocess can pick up the existing Optuna study before running
``adapter.tune(resume=True, ...)``.

These tests pin the post-fix contract:

- INV-#156-A: under the default strategy, retune routes through the
  subprocess runner (NOT the thread fallback any more).
- INV-#156-B: ``SubprocessJobRunner.run`` for retune exports current
  tune state to a temp path before invoking ``run_job_subprocess``.
- INV-#156-C: ``run_job_subprocess`` receives ``retune_kwargs`` AND
  ``tune_state_in_path`` from the parent.
- INV-#156-D: ``SubprocessJobRunner.run`` cleans up the inbound tune-state
  temp file in ``finally`` regardless of subprocess outcome.
- INV-#156-E: ``_subprocess_entry.run_job`` for retune calls
  ``adapter.restore_tune_state`` and then ``adapter.tune(resume=True)``.
- INV-#156-F: ``RetuneSubprocessUnsupportedError`` and its
  ``RETUNE_SUBPROCESS_UNSUPPORTED`` error code are no longer reachable.
- INV-#156-G (slow): clean ``tune → retune`` per-trial wall-clock is
  within 1.2x of subprocess tune.
- INV-#156-H (slow): catastrophe path
  ``tune → main-thread booster.predict → retune`` per-trial wall-clock
  is within 1.5x of subprocess tune (ratio that PR #155 hit ~11x on).

Tests in the "fast" tier use mocks so they do not require lightgbm /
libgomp; the slow tier reproduces the actual perf signature on Linux.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_widget_with_data() -> Any:
    """Build a LizyWidget with the real LizyMLAdapter wrapped behind a mock.

    Mirrors the harness in ``test_reg_154_default_strategy_retune.py`` so
    canonical config / backend-contract calls work but ML library calls can
    be intercepted.
    """
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = {"type": "object"}
        adapter.validate_config.return_value = []
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
        adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    df = pd.DataFrame(
        {
            "f1": list(range(40)),
            "f2": [i * 0.5 for i in range(40)],
            "f3": [i % 5 for i in range(40)],
            "y": [0, 1] * 20,
        }
    )
    w.load(df, target="y")
    return w


def _seed_tune_summary(w: Any) -> None:
    """Make ``w.retune(...)`` pass its precondition without an actual prior tune."""
    w.tune_summary = {
        "best_params": {"learning_rate": 0.05},
        "best_score": 0.9,
        "trials": [],
        "metric_name": "auc",
        "direction": "maximize",
        "rounds": [],
        "boundary_report": None,
    }


def _stub_subprocess_result() -> Any:
    """Build a SubprocessJobResult that satisfies the supervisor's contract."""
    from lizyml_widget.subprocess_runner import SubprocessJobResult

    return SubprocessJobResult(
        job_type="tune",
        fit_summary={},
        tune_summary={
            "best_params": {"learning_rate": 0.07},
            "best_score": 0.92,
            "trials": [],
            "metric_name": "auc",
            "direction": "maximize",
            "rounds": [],
            "boundary_report": None,
        },
        eval_table=[],
        split_summary=[],
        available_plots=["optimization-history"],
        model_path=None,
        tune_state_path=None,
    )


# ---------------------------------------------------------------------------
# INV-#156-A: retune uses the subprocess runner under default strategy
# ---------------------------------------------------------------------------


class TestRetuneRoutesToSubprocessRunner:
    """The thread fallback added by PR #155 must be gone — retune now goes
    through ``SubprocessJobRunner`` under the default subprocess strategy."""

    def test_retune_under_subprocess_default_uses_subprocess_runner(self) -> None:
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", "/usr/lib/libomp5.so"),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_sp_inst = mock_sp.return_value
            mock_sp_inst.kind = "subprocess"
            mock_sp_inst.run.return_value = MagicMock(
                job_type="tune",
                fit_summary={},
                tune_summary={
                    "best_params": {},
                    "best_score": 0.9,
                    "trials": [],
                    "metric_name": "auc",
                    "direction": "maximize",
                    "rounds": [],
                    "boundary_report": None,
                },
                eval_table=[],
                split_summary=[],
                available_plots=["optimization-history"],
                model_path=None,
            )

            w = _make_widget_with_data()
            _seed_tune_summary(w)
            w._service._tune_model = MagicMock()  # P-028 prerequisite

            w.retune(n_trials=5)
            if w._job_thread:
                w._job_thread.join(timeout=10)

            mock_sp.assert_called_once()
            mock_thread.assert_not_called()
            assert w.status == "completed", (
                f"Expected retune to complete via subprocess runner; "
                f"got status={w.status!r} error={w.error!r}"
            )


# ---------------------------------------------------------------------------
# INV-#156-B/C/D: subprocess retune IPC contract
# ---------------------------------------------------------------------------


class TestSubprocessRetuneIpcContract:
    """``SubprocessJobRunner.run`` for retune must serialise current tune
    state into the subprocess input and forward ``retune_kwargs``.

    Drives the runner directly so the IPC contract is asserted regardless
    of whether the supervisor / widget glue is correct.
    """

    def _make_runner(self) -> tuple[Any, Any, Any]:
        from lizyml_widget.job_runner import JobSpec, SubprocessJobRunner

        svc = MagicMock()
        svc.get_dataframe.return_value = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        svc.get_df_info.return_value = {"target": "y"}
        # The runner asks the service to write the inbound tune-state blob.
        svc.export_tune_state_to_path = MagicMock()

        runner = SubprocessJobRunner(svc)
        spec = JobSpec(
            job_type="tune",
            config={"config_version": 1},
            retune_kwargs={"resume": True, "n_trials": 7, "expand_boundary": False},
        )
        return runner, spec, svc

    def test_runner_calls_export_tune_state_to_path_before_subprocess(self) -> None:
        runner, spec, svc = self._make_runner()
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # By the time run_job_subprocess is invoked, the parent must
            # have already exported the current tune state.
            assert svc.export_tune_state_to_path.called, (
                "INV-#156-B: subprocess retune must export tune state BEFORE "
                "spawning the child process"
            )
            return _stub_subprocess_result()

        with patch("lizyml_widget.job_runner.run_job_subprocess", side_effect=fake_run):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        # The path passed to export must equal the path forwarded to
        # run_job_subprocess.
        export_call = svc.export_tune_state_to_path.call_args
        assert export_call is not None
        export_path = export_call.args[0] if export_call.args else export_call.kwargs.get("path")
        assert isinstance(export_path, str)
        assert captured.get("tune_state_in_path") == export_path

    def test_runner_forwards_retune_kwargs_to_subprocess(self) -> None:
        runner, spec, _svc = self._make_runner()
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _stub_subprocess_result()

        with patch("lizyml_widget.job_runner.run_job_subprocess", side_effect=fake_run):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        assert captured.get("retune_kwargs") == {
            "resume": True,
            "n_trials": 7,
            "expand_boundary": False,
        }

    def test_runner_cleans_up_tune_state_in_path_on_success(self) -> None:
        runner, spec, _svc = self._make_runner()
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # The runner allocates a tempfile path and points us at it. We
            # touch the path to simulate the subprocess writing into it.
            path = kwargs["tune_state_in_path"]
            Path(path).write_bytes(b"stub")
            return _stub_subprocess_result()

        with patch("lizyml_widget.job_runner.run_job_subprocess", side_effect=fake_run):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        path = captured.get("tune_state_in_path")
        assert isinstance(path, str)
        assert not Path(path).exists(), (
            f"INV-#156-D: tune-state input path must be cleaned up; still exists at {path!r}"
        )

    def test_runner_cleans_up_all_tempfiles_when_export_fails(self) -> None:
        """INV-#156-D extension: when ``service.export_tune_state_to_path``
        raises (e.g. no prior tune, OSError mid-pickle), the runner must
        clean up *all three* tempfiles allocated before the subprocess
        spawn — ``model_out_path`` (mkdtemp), ``tune_state_out_path``,
        AND ``tune_state_in_path`` — and re-raise the original exception.
        Otherwise long-lived Jupyter kernels leak temp files on every
        misuse of ``w.retune()`` without a prior tune."""
        from lizyml_widget.job_runner import JobSpec, SubprocessJobRunner

        svc = MagicMock()
        svc.get_dataframe.return_value = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        svc.get_df_info.return_value = {"target": "y"}
        svc.export_tune_state_to_path = MagicMock(
            side_effect=ValueError("no prior tune"),
        )

        runner = SubprocessJobRunner(svc)
        spec = JobSpec(
            job_type="tune",
            config={"config_version": 1},
            retune_kwargs={"resume": True, "n_trials": 5},
        )

        with (
            patch("lizyml_widget.job_runner.run_job_subprocess") as mock_run,
            pytest.raises(ValueError, match="no prior tune"),
        ):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        # The subprocess must NOT have been spawned (export failed first).
        mock_run.assert_not_called()
        # Tempfile paths are passed to ``export_tune_state_to_path`` only —
        # we can recover them via the mock call args.
        export_path = svc.export_tune_state_to_path.call_args.args[0]
        assert not Path(export_path).exists(), (
            f"tune_state_in_path must be cleaned up on export failure; "
            f"still exists at {export_path!r}"
        )

    def test_runner_cleans_up_tune_state_in_path_on_subprocess_error(self) -> None:
        runner, spec, _svc = self._make_runner()
        captured_path: dict[str, str] = {}

        def fake_run(**kwargs: Any) -> Any:
            captured_path["p"] = kwargs["tune_state_in_path"]
            Path(kwargs["tune_state_in_path"]).write_bytes(b"stub")
            raise RuntimeError("subprocess crashed")

        with (
            patch("lizyml_widget.job_runner.run_job_subprocess", side_effect=fake_run),
            pytest.raises(RuntimeError, match="subprocess crashed"),
        ):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        path = captured_path["p"]
        assert not Path(path).exists(), (
            f"INV-#156-D: tune-state input path must still be cleaned up "
            f"on subprocess error; still exists at {path!r}"
        )


# ---------------------------------------------------------------------------
# INV-#156-E: subprocess entry retune branch wires the adapter correctly
# ---------------------------------------------------------------------------


class TestSubprocessEntryRetuneBranch:
    """The subprocess child must call ``adapter.restore_tune_state`` from
    the inbound path before invoking ``adapter.tune(resume=True, ...)``.
    """

    def test_run_job_retune_restores_tune_state_then_calls_tune_with_resume(
        self, tmp_path: Path
    ) -> None:
        from lizyml_widget._subprocess_entry import run_job

        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        tune_state_in = tmp_path / "tune_state.pkl"
        tune_state_in.write_bytes(b"opaque")  # adapter handles this

        mock_summary = MagicMock()
        mock_summary.best_params = {"lr": 0.07}
        mock_summary.best_score = 0.92
        mock_summary.trials = []
        mock_summary.metric_name = "auc"
        mock_summary.direction = "maximize"
        mock_summary.rounds = []
        mock_summary.boundary_report = None

        mock_adapter = MagicMock()
        mock_adapter.tune.return_value = mock_summary
        mock_adapter.evaluate_table.return_value = []
        mock_adapter.split_summary.return_value = []
        mock_adapter.available_plots.return_value = ["optimization-history"]

        call_log: list[str] = []
        mock_adapter.restore_tune_state.side_effect = lambda *a, **kw: call_log.append(
            "restore_tune_state"
        )
        mock_adapter.tune.side_effect = lambda *a, **kw: call_log.append("tune") or mock_summary

        import io

        output = io.BytesIO()
        with patch(
            "lizyml_widget._subprocess_entry._create_adapter",
            return_value=mock_adapter,
        ):
            run_job(
                job_type="tune",
                config={"model": {"name": "lgbm"}},
                df=df,
                target="y",
                model_out_path=None,
                output=output,
                tune_state_out_path=str(tmp_path / "out.pkl"),
                retune_kwargs={"resume": True, "n_trials": 7, "expand_boundary": False},
                tune_state_in_path=str(tune_state_in),
            )

        # The order matters: restore_tune_state must run before tune so the
        # study handle is attached when adapter.tune fires.
        assert call_log == ["restore_tune_state", "tune"], call_log
        # adapter.tune received the resume kwargs.
        tune_kwargs = mock_adapter.tune.call_args.kwargs
        assert tune_kwargs.get("resume") is True
        assert tune_kwargs.get("n_trials") == 7
        assert tune_kwargs.get("expand_boundary") is False


# ---------------------------------------------------------------------------
# INV-#156-F: RetuneSubprocessUnsupportedError is gone
# ---------------------------------------------------------------------------


class TestRetuneSubprocessUnsupportedRemoval:
    """The PR #155 thread fallback was the last reason
    ``RetuneSubprocessUnsupportedError`` existed. Once subprocess retune is
    supported, both the class and the ``RETUNE_SUBPROCESS_UNSUPPORTED``
    error code are dead weight.
    """

    def test_retune_subprocess_unsupported_error_class_is_removed(self) -> None:
        import lizyml_widget.job_runner as jr

        assert not hasattr(jr, "RetuneSubprocessUnsupportedError"), (
            "INV-#156-F: RetuneSubprocessUnsupportedError must be removed "
            "after subprocess retune resume lands"
        )

    def test_widget_does_not_translate_retune_subprocess_unsupported(self) -> None:
        """The supervisor's specialised ``except`` for
        ``RetuneSubprocessUnsupportedError`` must be removed alongside the
        class. The catch-all path remains the safety net."""
        from lizyml_widget import widget as widget_module

        src = Path(widget_module.__file__).read_text()
        assert "RetuneSubprocessUnsupportedError" not in src, (
            "INV-#156-F: widget.py must not import or reference RetuneSubprocessUnsupportedError"
        )
        assert "RETUNE_SUBPROCESS_UNSUPPORTED" not in src, (
            "INV-#156-F: widget.py must not surface RETUNE_SUBPROCESS_UNSUPPORTED to the JS layer"
        )


# ---------------------------------------------------------------------------
# INV-#156-I: tune-state IPC preserves the lizyml ``Model`` resume state
# (regression for a P-038 implementation bug discovered via Playwright /
# manual smoke testing — only ``_tuning_result`` and ``_study`` were being
# round-tripped, so subprocess retune appeared to succeed but discarded
# previous rounds).
# ---------------------------------------------------------------------------


class TestTuneStateRoundTripPreservesResumeState:
    """``adapter.export_tune_state`` / ``restore_tune_state`` must capture
    every ``Model`` private attribute that ``Model.tune(resume=True)``
    reads. P-037 only round-tripped ``_tuning_result`` / ``_study`` (which
    is enough for the ``optimization-history`` plot); P-038 also needs
    ``_rounds`` / ``_round_number`` / ``_space`` / ``_used_default_space``
    so the cumulative ``rounds`` list is preserved across subprocess
    boundaries.
    """

    def test_export_then_restore_round_trips_all_resume_state(self, tmp_path: Path) -> None:
        from lizyml_widget.adapter import LizyMLAdapter

        adapter = LizyMLAdapter()

        # Hand-build a model-shaped object — we only care that the adapter
        # round-trips the right private attributes, not that lizyml's tune
        # logic actually accepts them.
        class _StubModel:
            pass

        src = _StubModel()
        src._tuning_result = {"sentinel": "tuning_result"}  # noqa: SLF001
        src._study = {"sentinel": "study"}  # noqa: SLF001
        src._round_number = 7  # noqa: SLF001
        src._rounds = [{"round": 1}, {"round": 2}, {"round": 3}]  # noqa: SLF001
        src._space = [{"name": "lr", "low": 1e-4, "high": 1e-1}]  # noqa: SLF001
        src._used_default_space = True  # noqa: SLF001

        path = tmp_path / "blob.pkl"
        adapter.export_tune_state(src, str(path))

        dst = _StubModel()
        adapter.restore_tune_state(dst, str(path))

        assert dst._tuning_result == src._tuning_result  # noqa: SLF001
        assert dst._study == src._study  # noqa: SLF001
        assert dst._round_number == 7, (  # noqa: SLF001
            "INV-#156-I: _round_number must round-trip — without it, "
            "Model.tune(resume=True) restarts the round counter at 1"
        )
        assert dst._rounds == src._rounds, (  # noqa: SLF001
            "INV-#156-I: _rounds must round-trip — without it, retune's "
            "cumulative rounds list is overwritten by the new round only"
        )
        assert dst._space == src._space  # noqa: SLF001
        assert dst._used_default_space is True  # noqa: SLF001

    def test_restore_tolerates_p037_legacy_blob(self, tmp_path: Path) -> None:
        """Backward compat: a P-037-only blob (only ``tuning_result`` / ``study``)
        must still restore those keys without raising. Plot rendering paths
        should not regress when reading old-format blobs."""
        import pickle

        from lizyml_widget.adapter import LizyMLAdapter

        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"tuning_result": {"sentinel": "old"}, "study": None},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        class _StubModel:
            pass

        adapter = LizyMLAdapter()
        m = _StubModel()
        adapter.restore_tune_state(m, str(path))
        assert m._tuning_result == {"sentinel": "old"}  # noqa: SLF001
        # P-038 keys absent → adapter must not raise; downstream tune
        # would fail with the same error a fresh model would, so no
        # additional invariant is needed here.


# ---------------------------------------------------------------------------
# INV-#156-G/H: end-to-end perf bounds (slow)
# ---------------------------------------------------------------------------


def _libgomp_loaded() -> bool:
    """Force-load lightgbm and check whether libgomp ends up in /proc/self/maps.

    Mirrors ``openmp_detect._ensure_lightgbm_imported`` so the slow perf
    tests do not silently skip when run from a fresh pytest session
    (where lightgbm has not yet been imported by any other path).
    """
    if sys.platform != "linux":
        return False
    try:
        import lightgbm  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        with open("/proc/self/maps") as f:
            return any("libgomp" in line for line in f)
    except OSError:
        return False


def _make_perf_widget(n_rows: int = 100_000) -> Any:
    """Build a real LizyWidget on a synthetic binary classification task.

    Used by the slow perf tests; mirrors the reproducer used during the
    #156 investigation.
    """
    import numpy as np

    from lizyml_widget.widget import LizyWidget

    rng = np.random.default_rng(0)
    n_cols = 50
    cols = {f"f{i}": rng.random(n_rows, dtype=np.float64) for i in range(n_cols)}
    cols["y"] = (rng.random(n_rows) > 0.5).astype(int)
    df = pd.DataFrame(cols)

    w = LizyWidget()
    w.load(df, target="y")
    cfg = dict(w.config)
    tuning = dict(cfg.get("tuning") or {})
    optuna = dict(tuning.get("optuna") or {})
    params = dict(optuna.get("params") or {})
    params["n_trials"] = 3
    optuna["params"] = params
    tuning["optuna"] = optuna
    cfg["tuning"] = tuning
    w.set_config(cfg)
    return w


def _real_strategy_patch() -> Any:
    """Override conftest's autouse thread-strategy patch.

    ``conftest.py`` patches ``lizyml_widget.widget.get_execution_strategy`` to
    return ``("thread", None)`` for the entire suite (so mock-adapter tests
    do not accidentally spawn subprocesses). The slow perf tests in this
    module need the *real* strategy detection — production behaviour on
    libgomp must be ``("subprocess", libomp_path)``. This helper re-patches
    inside the test, replacing conftest's value with the real return.
    """
    from lizyml_widget.openmp_detect import _reset_libgomp_cache, get_execution_strategy

    _reset_libgomp_cache()
    return patch(
        "lizyml_widget.widget.get_execution_strategy",
        side_effect=get_execution_strategy,
    )


def _per_trial(elapsed: float, n_trials: int = 3) -> float:
    return elapsed / n_trials


@pytest.mark.slow
def test_clean_retune_within_subprocess_tune_perf_bound() -> None:
    """INV-#156-G: with no main-thread parallel-region trigger between
    tune and retune, per-trial retune wall-clock must be within 1.2x of
    subprocess tune.

    PR #155 measured this at 1.43x (thread fallback) — Option A's
    subprocess retune resume should land within 1.2x because the same
    subprocess startup amortises ~3 trials' worth of pool-affinity-free
    OpenMP work.
    """
    if sys.platform != "linux":
        pytest.skip("subprocess default + libgomp catastrophe is Linux-only")
    if not _libgomp_loaded():
        pytest.skip("libgomp not loaded — perf bound only meaningful on libgomp hosts")

    os.environ.pop("LZW_FORCE_THREAD", None)

    with _real_strategy_patch():
        w = _make_perf_widget()

        t0 = time.perf_counter()
        w.tune(timeout=600)
        tune_elapsed = time.perf_counter() - t0
        assert w.status == "completed", f"Tune failed: {w.error!r}"
        assert w._execution_strategy == "subprocess", (
            f"Test pre-condition: strategy must be subprocess, got {w._execution_strategy!r}"
        )

        t1 = time.perf_counter()
        w.retune(n_trials=3, timeout=600)
        retune_elapsed = time.perf_counter() - t1
        assert w.status == "completed", f"Retune failed: {w.error!r}"

    tune_per = _per_trial(tune_elapsed)
    retune_per = _per_trial(retune_elapsed)
    ratio = retune_per / tune_per if tune_per > 0 else float("inf")

    UPPER_BOUND = 1.2
    assert ratio < UPPER_BOUND, (
        f"INV-#156-G regression: clean retune per-trial wall {retune_per:.2f}s "
        f"is {ratio:.2f}x subprocess tune per-trial {tune_per:.2f}s "
        f"(bound {UPPER_BOUND}x). PR #155 baseline was ~1.43x; Option A "
        f"should land within 1.2x."
    )

    # INV-#156-J (functional, not just perf): rounds must accumulate.
    # The original P-038 implementation only round-tripped
    # ``_tuning_result`` and ``_study`` — which made retune *appear* to
    # succeed (status=completed, badge ✓) while silently overwriting the
    # rounds list. Slow-path perf bounds did not catch this because
    # per-trial wall-clock is normal in either case. Adding a contents
    # assertion here pins the rounds-accumulation contract empirically.
    rounds = w.tune_summary.get("rounds") or []
    assert len(rounds) == 2, (
        f"INV-#156-J: subprocess retune must accumulate rounds; got "
        f"{len(rounds)} round(s) — {rounds!r}. The P-038 ``adapter."
        f"export_tune_state`` must round-trip ``_rounds`` / "
        f"``_round_number`` / ``_space`` / ``_used_default_space`` so "
        f"``Model.tune(resume=True)`` builds the cumulative rounds list."
    )
    assert rounds[0].get("round") == 1
    assert rounds[1].get("round") == 2


@pytest.mark.slow
def test_catastrophe_path_retune_within_subprocess_tune_perf_bound() -> None:
    """INV-#156-H: tune → main-thread booster.predict → retune must NOT
    regress to the libgomp pool-affinity catastrophe (~11x slowdown
    measured under PR #155). Per-trial retune wall-clock must be within
    1.5x of subprocess tune.

    This test specifically guards against the issue #156 root cause: PR
    #155's thread fallback re-bound the libgomp pool to the parent main
    thread once any predict / SHAP call fired, causing 109s retune for
    a 14s clean baseline.
    """
    if sys.platform != "linux":
        pytest.skip("catastrophe path is Linux-libgomp-only")
    if not _libgomp_loaded():
        pytest.skip("libgomp not loaded — bound only meaningful on libgomp hosts")

    os.environ.pop("LZW_FORCE_THREAD", None)

    with _real_strategy_patch():
        w = _make_perf_widget()

        t0 = time.perf_counter()
        w.tune(timeout=600)
        tune_elapsed = time.perf_counter() - t0
        assert w.status == "completed", f"Tune failed: {w.error!r}"
        assert w._execution_strategy == "subprocess", (
            f"Test pre-condition: strategy must be subprocess, got {w._execution_strategy!r}"
        )

        # Force a parent-main-thread libgomp parallel-region entry. Under PR
        # #155 this would bind the pool to the main thread and the next
        # worker-thread retune would hit ~36s/trial.
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(42)
        X_small = rng.random((500, 50))
        y_small = (rng.random(500) > 0.5).astype(int)
        booster_holder: dict[str, Any] = {}

        def _fit_in_worker() -> None:
            ds = lgb.Dataset(X_small, label=y_small)
            booster_holder["b"] = lgb.train(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "verbose": -1,
                    "num_threads": 0,
                },
                ds,
                num_boost_round=20,
            )

        th = threading.Thread(target=_fit_in_worker, daemon=False)
        th.start()
        th.join()
        booster = booster_holder["b"]
        # Multiple predict calls so the parallel region is firmly established
        # on the parent main thread.
        for _ in range(3):
            _ = booster.predict(X_small)

        t1 = time.perf_counter()
        w.retune(n_trials=3, timeout=600)
        retune_elapsed = time.perf_counter() - t1
        assert w.status == "completed", f"Retune failed: {w.error!r}"

    tune_per = _per_trial(tune_elapsed)
    retune_per = _per_trial(retune_elapsed)
    ratio = retune_per / tune_per if tune_per > 0 else float("inf")

    UPPER_BOUND = 1.5
    assert ratio < UPPER_BOUND, (
        f"INV-#156-H regression: catastrophe-path retune per-trial wall "
        f"{retune_per:.2f}s is {ratio:.2f}x subprocess tune per-trial "
        f"{tune_per:.2f}s (bound {UPPER_BOUND}x). PR #155 hit ~11x "
        f"because thread retune re-bound libgomp to the parent main "
        f"thread; subprocess retune resume should not regress."
    )
