"""BackendExecutor — caller-thread ML library call funnel (P-039 Phase 3).

Every caller-thread invocation of an ML library function (LightGBM /
SHAP / etc.) goes through ``BackendExecutor.run_ml``. The executor:

1. Centralises ``WidgetService._libgomp_pool_owner`` marking so the INV-G
   runtime guard (P-039 Phase 2, ``widget._run_job``) sees a single,
   consistent "main thread bound libgomp" signal regardless of which
   adapter method was called.
2. Provides a single chokepoint for a future scoped routing decision
   (subprocess offload, dedicated OpenMP-owner thread). Phase 3 runs
   everything inline on the caller thread; Phase 4 can then enforce
   "no direct ML library call outside the executor" via lint rule
   without renaming any call sites.

The runner-side path (``ThreadJobRunner`` / ``SubprocessJobRunner``) is
*not* funnelled through the executor — runners are the equivalent
chokepoint for worker-thread / subprocess ML, and ``widget._supervise``
already calls ``service.mark_libgomp_owner`` after a successful
``runner.run()``. The executor is only for code that runs *on the same
thread as the user-facing call* (predict, SHAP / explainability plots).

State transitions performed by the executor:

- ``ml_kind="predict"`` → mark ``"main"`` (OpenMP parallel boost in
  ``booster.predict`` / etc. binds to caller thread).
- ``ml_kind="explain"`` / ``ml_kind="plot_shap"`` → mark ``"main"``
  (SHAP TreeExplainer is parallel under OpenMP).
- ``ml_kind="plot_other"`` → no state change (Plotly-only plot
  computation does not enter ML library parallel regions).

Marking happens in a ``finally`` block so that exceptions raised mid-
parallel-region still update the state defensively (libgomp may have
already bound by the time the exception unwinds).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from .service import WidgetService


MLKind = Literal["predict", "explain", "plot_shap", "plot_other"]
"""Categories of caller-thread ML library calls.

- ``predict``: model inference (LightGBM ``booster.predict`` and similar)
- ``explain``: SHAP value computation (``TreeExplainer.shap_values``)
- ``plot_shap``: SHAP-based plots (``importance_plot(kind='shap')``)
- ``plot_other``: non-ML-library plots (learning curves,
  optimization-history, ROC, etc. — pure plotly / numpy work)
"""

# Kinds that enter an OpenMP parallel region on the caller thread and
# therefore bind libgomp's pool affinity. Anything in this set must
# transition the libgomp owner state to ``"main"``.
_KINDS_THAT_BIND_LIBGOMP: frozenset[MLKind] = frozenset({"predict", "explain", "plot_shap"})

T = TypeVar("T")


class BackendExecutor:
    """Funnel for caller-thread ML library calls.

    Constructed once per ``WidgetService`` instance and held as
    ``service._executor``. All of ``service.predict``,
    ``service.get_plot``, and ``service.get_inference_plot`` route
    their adapter calls through ``run_ml`` so the libgomp owner state
    is updated in exactly one place.
    """

    def __init__(self, service: WidgetService) -> None:
        self._service = service

    def run_ml(self, op: Callable[[], T], *, ml_kind: MLKind) -> T:
        """Run an ML library operation on the caller thread.

        Parameters
        ----------
        op:
            Zero-arg callable that performs the ML library call (e.g.
            ``lambda: adapter.predict(model, df)``).
        ml_kind:
            Category of the call. Determines whether to mark the
            libgomp owner state to ``"main"`` after the call.

        Returns
        -------
        The result of ``op()``.

        Raises
        ------
        Whatever ``op()`` raises. State marking still happens on the
        exception path (libgomp may have already entered a parallel
        region before the exception was raised).
        """
        try:
            return op()
        finally:
            if ml_kind in _KINDS_THAT_BIND_LIBGOMP:
                self._service.mark_libgomp_owner("main")
