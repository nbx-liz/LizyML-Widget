# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **ConfigTab debounce / traitlet sync race**
  ([#136](https://github.com/nbx-liz/LizyML-Widget/issues/136)).
  When Python pushed a new ``config`` (e.g. after ``apply_best_params``)
  while a user edit was still mid-debounce, the pending timer fired
  later and computed a patch against a stale baseline, silently
  overwriting Python's update. The ``[config]`` ``useEffect`` now
  cancels the pending debounce timer before resyncing local state.

### Added
- **State-machine invariants declared (INV-A..F)** (P-033,
  [#118](https://github.com/nbx-liz/LizyML-Widget/issues/118)).
  `BLUEPRINT.md` §6.4 now enumerates six invariants over the widget's
  status FSM, job-thread singleton, `_tune_model` ownership,
  `_cancel_flag` lifecycle, `progress.round` monotonicity, and
  `boundary_report.dims` uniqueness. Each invariant has a
  RED-then-GREEN test in `tests/test_invariants.py`, and the relevant
  guards in `widget.py::_run_job` / `_supervise` carry inline INV-X
  breadcrumbs so future PRs reviewing those sites see the contract
  they must preserve.

### Changed
- **JobRunner Protocol extracted from widget.py** (P-032,
  [#117](https://github.com/nbx-liz/LizyML-Widget/issues/117)).
  The widget no longer carries two near-duplicate `_job_worker` methods
  for the in-process and subprocess execution paths. A new
  `src/lizyml_widget/job_runner.py` defines the `JobRunner` Protocol
  with `ThreadJobRunner` / `SubprocessJobRunner` implementations, and
  `widget.py::_supervise` owns all state-machine transitions, traitlet
  plumbing, and error classification for both runners. The legacy
  `_job_worker` and `_subprocess_job_worker` are removed; `JobSpec`
  carries `job_type` / `config` / `retune_kwargs` immutably.
  `RETUNE_SUBPROCESS_UNSUPPORTED` is now a typed
  `RetuneSubprocessUnsupportedError` raised by `SubprocessJobRunner`
  and translated to the widget error code in `_supervise`. New
  `tests/test_job_runner.py` covers each runner across normal
  completion / cancel / exception / retune-rejection.
- **Adapter boundary typed via `LMResultView`** ([#116](https://github.com/nbx-liz/LizyML-Widget/issues/116)).
  All `getattr(...)` reads on lizyml result objects are consolidated into a
  new `adapter_views.py` module (`view_fit_result`, `view_tuning_result`,
  `view_tune_progress`, `view_prediction_result`, `view_boundary_report`,
  `view_rounds`). Each view raises `LizyMLContractError` on a missing
  required field, giving the version guard real teeth — a renamed field
  in a future lizyml minor fails fast at the boundary rather than
  silently degrading to `None` / `[]` deeper in the widget.
- **Removed `model._widget_config` private write** ([#116](https://github.com/nbx-liz/LizyML-Widget/issues/116)).
  `LizyMLAdapter.create_model` no longer monkey-patches the lizyml model
  with a widget-specific attribute. Configs now live in an adapter-side
  registry keyed by `id(model)`, and the legacy `_cfg.task` / fallback
  config read is centralised in a single `_task_for_model(model)`
  helper used by both `available_plots()` and `model_info()`.

### Added
- **E2E test coverage Phase B** ([#114](https://github.com/nbx-liz/LizyML-Widget/issues/114)).
  Four new test files plus a Tune→Apply→Re-tune extension to the existing
  user-flow suite:
  - `test_p030_compat.py` — multiclass string-label round-trip + smape/wape
    regression chip rendering (locks in P-030 acceptance).
  - `test_inference_flow.py` — Fit → Open Inference → Run Inference →
    PredTable rows, plus SHAP-toggle dispatch.
  - `test_error_flows.py` — backend-error banner + Re-run gate.
  - `test_user_flows.py` — Tune → Apply to Fit → Re-tune resume → Boundary
    Expansion panel.
  Two new notebooks (`test_multiclass_strings.ipynb`,
  `test_regression_smape_wape.ipynb`) drive the P-030 fixtures, with
  matching `multiclass_widget_page` / `regression_smape_wape_page` Playwright
  fixtures in `conftest.py`.
- **JS test coverage Phase A** ([#114](https://github.com/nbx-liz/LizyML-Widget/issues/114)).
  Vitest suite expanded from 171 to 272 cases covering `App.tsx`, `Header.tsx`,
  `configHelpers.ts`, `ModelEditors.tsx`, `TuneSubTab.tsx`, `FitSubTab.tsx`,
  `DistributionBar.tsx`, `FoldPreview.tsx`, `PredTable.tsx`,
  `BlockedGroupKFold.tsx`, plus the P-030 smape/wape regression chip in
  `SearchSpace.tsx` and failed/error rendering paths in `ResultsTab.tsx`.
  Statement coverage rose from 47% to 75%.
- `pnpm test:coverage` script and `vitest.config.ts` `coverage.thresholds`
  block (75% statements/lines, 70% branches, 50% functions). CI now runs
  `pnpm test:coverage` and prints the e2e test count for visibility.
- **Backend contract**: new `cv_strategy_labels` and `additional_params_hidden_keys`
  capabilities ([#119](https://github.com/nbx-liz/LizyML-Widget/issues/119)).
  Adding a new CV strategy in `adapter_contract.py` now surfaces in the UI dropdown
  without any JS edit; a `humaniseSnake()` fallback covers labels missing from the map.

### Changed
- **Code quality**: Closed two HIGH-tier code-review findings ([#115](https://github.com/nbx-liz/LizyML-Widget/issues/115)).
  - `LizyWidget.model_info` now routes through a new `WidgetService.model_info(model)`
    delegate instead of reaching into `Service._adapter` private state.
  - Bare `except Exception: pass` blocks in `WidgetService._default_strategy_for_task`,
    `WidgetService._default_cv_state`, and `_job_worker`'s tune-only fit-summary
    fallback have been narrowed to documented exception types and now leave a
    `_log.debug` breadcrumb instead of swallowing silently.
- **JS no longer hardcodes LightGBM-specific catalogs** ([#119](https://github.com/nbx-liz/LizyML-Widget/issues/119)).
  - `DataTab.tsx`: dropped the `CV_STRATEGIES` literal — strategy chips now derive
    from `backend_contract.capabilities.cv_strategies`/`cv_strategy_labels`. The
    `cv.strategy === "kfold"` and `cv.strategy === "blocked_group_kfold"`
    equality literals are replaced with capability-driven checks.
  - `FitSubTab.tsx`: `GROUP_STRATEGIES`/`TIME_STRATEGIES` literals replaced with
    `cv_strategy_fields`-driven derivation.
  - `ModelEditors.tsx`: `HANDLED_MODEL_FIELDS` static set is gone — derived from
    `search_space_catalog` (smart_params group) plus structural keys. The
    `num_leaves ?? 256` defaults are removed; default flows from the catalog.
    The `verbose / num_threads` exclusion comes from
    `additional_params_hidden_keys`.

## [0.9.0] - 2026-05-08

### Changed
- **Required lizyml version bumped to `>=0.10.0,<0.13`** (P-030, [#112](https://github.com/nbx-liz/LizyML-Widget/issues/112))
  — the widget now admits lizyml 0.10 / 0.11 / 0.12. Lower bound is raised to
  0.10.0 because the Adapter relies on lizyml 0.10's `FitResult.target_encoder`
  for label dtype preservation; running the new code paths against 0.9.x would
  surface late-bound `AttributeError`. Existing users on lizyml 0.9.x must
  upgrade alongside the widget.

### Added
- **Non-numeric classification target round-trip** (P-030) — lizyml 0.10
  auto-encodes non-numeric `y` (`object` / `pd.StringDtype` / `category` / `bool`)
  via `TargetEncoder` and decodes predictions back to the original label dtype.
  The widget now passes that contract through transparently: `LizyWidget.predict()`
  on a multiclass model trained on string labels returns `pred` values like
  `"Adelie"` / `"Chinstrap"` rather than int codes. New regression test
  `test_reg_112_target_encoder_roundtrip.py` locks this in.
- **smape / wape regression metrics** (P-030) — lizyml 0.11's zero-tolerant
  percentage-style regression metrics are now exposed in the BackendContract
  `model_metric.regression` option set, surfaced as Search Space / Model tab
  metric chips, and routed correctly by tune direction resolution
  (`MODEL_METRIC_TO_EVAL` identity mappings + `minimize` direction).

### Compatibility
- The widget's compat-matrix doc (`docs/VERSION_COMPAT.md`) gains a new top
  row pinning `lizyml-widget 0.9.x` to `lizyml >=0.10.0,<0.13`. Older widgets
  remain documented for past-release reference.
- The 0.12 resumable-tuning Optuna storage is **not** surfaced through the
  widget UI in this release — that exposure is tracked separately and will
  ship under a follow-up Proposal.

## [0.8.0] - 2026-04-12

### Added
- **Re-tune monitoring (P-027)** — round-aware Tune progress display,
  Boundary Expansion panel, Convergence Signal banner, and backend
  version guard that requires `lizyml>=0.9.0,<0.10`. The `progress`
  traitlet now carries optional `round`, `total_rounds`,
  `cumulative_trials`, `expanded_dims`, `latest_score`, `latest_state`,
  and `best_score` fields during Tune runs.
- **Re-tune launcher (P-028)** — new `w.retune(n_trials=..., expand_boundary=..., boundary_threshold=...)`
  Python API and a matching `retune` UI action. The Results tab gains
  a "Re-tune (resume)" button inside the Best Params accordion so
  users can resume the Optuna study with additional trials (and
  optionally widen boundaries) without leaving the widget.
- **Tuning History accordion (P-029)** — the Results tab now renders
  lizyml's `Model.tuning_plot()` figure in a dedicated "Tuning History"
  accordion on Tune completion, via the standard PlotViewer pipeline.
- `TuningSummary` gains `rounds: list[dict]` and
  `boundary_report: dict | None` fields, propagated through
  `LizyMLAdapter.tune()` and the `tune_summary` traitlet so the new UI
  components can render round-aware history.
- `docs/VERSION_COMPAT.md` — documents the widget ↔ lizyml
  compatibility matrix, supported upgrade paths, and install
  recommendations.
- `WidgetService` gains a dedicated `_tune_model` slot so an
  intervening `fit()` cannot clobber the Optuna study that the next
  `retune()` must resume.
- Closes [#101](https://github.com/nbx-liz/LizyML-Widget/issues/101).

### Changed
- **Required lizyml version bumped to `>=0.9.0,<0.10`** (previously
  `>=0.7.0`). The widget no longer works with lizyml 0.7.x / 0.8.x;
  use `pip install "lizyml-widget[lizyml]"` to let pip auto-resolve a
  compatible backend. `LizyMLAdapter.__init__` now validates the
  installed lizyml version at import time and raises a clear
  `ImportError` with an upgrade hint if the backend is out of range.
- `ResultsTab` layout on Tune completion:
  Best Params → RetuneControls → Convergence Signal →
  Boundary Expansion → Tuning History → (existing) Score / Plots / etc.
- `BackendAdapter.tune()` Protocol and `LizyMLAdapter.tune()` accept
  `resume`, `n_trials`, `expand_boundary`, and `boundary_threshold`
  kwargs (all default to their pre-P-028 values, so in-tree callers
  are unchanged).
- README adds a Re-tune usage section and a lizyml compatibility
  matrix.

### Removed
- `js/src/components/ScoreHistoryChart.tsx` — the widget-local Plotly
  duplicate of lizyml's `tuning_plot`. The P-029 refactor consolidates
  on the backend figure, dropping roughly 2.4 KB from the production
  bundle.

### Fixed
- **ConvergenceSignal** showed the literal 6-character string
  `\u2713` instead of a real checkmark glyph because the JSX text node
  was not wrapped in an expression.
- **ConvergenceSignal** "Round N" label was off by one after the
  third tune: `ResultsTab` was passing `lastRound.round + 1`, but
  lizyml's `RoundSummary.round` is already 1-indexed. The `+1` has
  been removed along with the now-redundant `>= 1` guard.
- `WidgetService.tune(resume=True)` now takes the `_tune_model` check
  inside the existing `_model_lock` section to close a TOCTOU window
  where an intervening `load()` could race with the check.
- `ResultsTab` Best Score row renders an em-dash placeholder instead
  of the literal string `"undefined"` when `tune_summary.best_score`
  is missing (defensive guard).

## [0.7.3] - 2026-04-08

### Changed
- Default `balanced` from False to True
- Default `training.seed` from 42 to 1120
- Default `calibration.method` from "platt" to "isotonic"
- Default `n_trials` from 50 to 10
- `max_bin` search space mode from Range to Choice with values [15, 63, 127, 255, 511, 1023]

### Added
- `default_range` support in search space catalog for `bagging_freq`, `lambda_l1`, `lambda_l2`, `min_data_in_bin_ratio`
- `default_choices` support in search space catalog for `max_bin`
- SearchSpace UI uses `default_range`/`default_choices` when toggling from Fixed to Range/Choice mode

## [0.7.2] - 2026-04-04

### Fixed
- Widget header, tabs, and Fit/Tune subtabs not staying visible when scrolling — added `max-height: max(80vh, 620px)` to `.lzw-root` to force internal scroll
- Tune Search Space: removed invalid "Choice" mode from `num_leaves` (integer param only needs Fixed/Range)
- Tune Search Space: Feature Weights toggle now shows column/weight editor when enabled (previously only toggle with no configuration)
- Calibration: removed deprecated N Splits field
- Calibration: `+ Add` now uses a select dropdown with predefined params instead of free-text input
- Calibration: params list is now method-dependent (isotonic has LightGBM params; platt/beta have none)
- Calibration: string params (objective, metric) render as text input instead of numeric stepper

### Changed
- Tune Search Space group order changed to Smart Params → Model Params → Training (matching Fit tab)
- `+ Add` button moved inside Model Params group (was at grid bottom)
- `first_metric_only` moved to appear directly after Metric in both Fit and Tune views
- Search Space group headers now visually distinguished with background color, border, and uppercase styling
- Upgraded lizyml dependency to v0.7.3

## [0.7.1] - 2026-04-02

### Fixed
- `prepare_tune_overrides` incorrectly stripped smart params (`auto_num_leaves`, `num_leaves_ratio`, etc.) from tune config — LizyML backend supports them during tuning
- `prepare_tune_overrides` did not set `first_metric_only=True` to match LizyML's `default_fixed_params`, causing early stopping behavior to differ between Tune and Fit
- `apply_best_params` replaced `inner_valid` with `None` when `validation_ratio` was present, forcing a different code path than Tune — now updates `inner_valid.ratio` in-place
- BLUEPRINT.md referenced non-existent function `resolve_smart_params_from_dict` with incorrect explanation for `feature_weights` Fixed-only constraint
- HISTORY.md P-014 incorrectly stated smart params are not used during tuning

## [0.7.0] - 2026-04-02

### Added
- Sticky header and tab bar — headers stay visible when scrolling long config forms
- Inner validation method filtering by column availability (group/time columns)
- Run button visibility improvements in Config tab subtab bar

### Changed
- Config tab subtab bar padding and layout refined

### Fixed
- `classify_best_params` misclassified `auto_num_leaves`, `feature_weights`, `balanced` as model params instead of smart params
- `prepare_tune_overrides` replaced entire `training` section instead of shallow-merging (lost `seed`)
- `apply_best_params` lost smart params and calibration when using run-config snapshot (now uses dual snapshot: run config + UI config)
- Optuna `best_params.metric` (single string) was not wrapped in list for LightGBM
- Tune direction not resolved when `evaluation.metrics` was empty (now falls back to `model.params.metric`)
- SearchSpace Fixed→Choice mode switch nested array values in choices for `metric` field
- SearchSpace Fixed→Choice for boolean fields initialized with only one value instead of `[true, false]`
- `useEffect` loop risk in ConfigTab

## [0.6.0] - 2026-03-28

### Added
- `w.load_model(path)` — load trained model from file for inference without re-fitting (P-024)
- `w.model_info` — property returning model metadata (loaded state, parameters)
- Learning Curve metric filter with precision_at_k support
- E2E test infrastructure: Playwright + pytest-playwright
- Component tests: DataTab, DynForm, SearchSpace, ConfigFooter, ProgressView
- JS tests integrated into CI pipeline

### Changed
- Backend Contract driven CV decoupling (P-025)
- Dependabot PRs now target `develop` branch

### Fixed
- Binary buffer for large plots + download fallback on Colab
- Redundant conditionals in widget.py and ConfigTab.tsx
- CI: added actions:write to auto-release workflow

## [0.5.0] - 2026-03-28

### Added
- `w.load_model(path)` — Load trained model from file for inference without re-fitting (P-024)
- `w.model_info` — Property returning model metadata (loaded state, parameters)
- `adapter.model_info(model)` — Model metadata extraction in LizyMLAdapter
- Inference plot: dynamic prediction column detection (no longer hardcoded to `pred`)
- JS test infrastructure: Vitest + @testing-library/preact + jsdom (114 JS tests)
- E2E test infrastructure: Playwright + pytest-playwright (10 E2E scenarios)
- Component tests: DataTab, DynForm, SearchSpace, ConfigFooter, ProgressView (51 tests)
- Hook tests: useJobPolling (Colab + non-Colab), usePlot, useModel, useTheme (63 tests)
- `pnpm test` and `pnpm lint` now run in CI
- Binary buffer plot transfer for large Plotly JSON (>800KB) on Colab (D-1)
- Download fallback: DataURL fallback when Blob URL is blocked by Colab sandbox (D-2)
- CV strategy metadata exposed in `backend_contract.capabilities` (P-025)
- `special_search_space_fields` in uiSchema for contract-driven SearchSpace rendering

### Changed
- Backend Contract driven CV decoupling: DataTab reads CV strategy fields from contract with fallback (P-025)
- Service CV defaults delegated to adapter contract (P-025)
- SearchSpace special field keys read from uiSchema instead of hardcoded (P-025)
- Export Code button styled with accent-outline for better visibility
- Apply to Fit button styled with primary color to prevent oversight after Tune
- `auto-release.yml` now has `actions:write` permission for PyPI workflow dispatch

### Fixed
- Redundant `if task:` guard in `_handle_set_task` after early return
- Redundant `|| status === "running"` in ConfigTab `canRun` expression

## [0.4.2] - 2026-03-27

### Fixed
- Consecutive jobs (Fit→Fit, Tune→Apply→Fit) now work correctly on Google Colab — polling restarts on `job_index` change (A-1)
- Polled state clears on completed/failed status transitions, preventing frozen UI after Tune (A-1 regression)
- Thread safety: `_job_lock` prevents TOCTOU race in `_run_job` (C-1)
- Thread safety: `_model_lock` protects model access during concurrent operations (C-2)
- Thread safety: `_tune_config_snapshot` reads protected by `_job_lock` (C-3)
- Plot responses echo `request_id` for stale-response filtering on rapid tab switching (B-1)
- Inference button disabled during execution to prevent double-click (B-2)
- YAML Export button disabled during export to prevent duplicate downloads (B-3)
- Config form disabled during job execution to prevent accidental edits (B-5)
- Action dispatch migrated from traitlet sync to msg:custom for Colab ipywidgets 7.x compatibility (P-023)

### Added
- JS test infrastructure: Vitest + @testing-library/preact + jsdom
- 60 JS tests covering hook state lifecycle, polling, plot caching, and component guards
- MockModel test utility for anywidget model interface
- Python poll state transition tests for consecutive jobs and Tune→Fit flow
- Thread safety tests (TOCTOU guard, model lock, snapshot protection)
- Plot request_id echo-back tests with backward compatibility
- Colab fit diagnostic notebook

### Changed
- Export Code button styled with accent-outline for better visibility
- Apply to Fit button styled with primary color to prevent oversight after Tune

## [0.4.1] - 2026-03-21

### Added
- CONTRIBUTING.md with development workflow and quality gates
- SECURITY.md with vulnerability reporting policy
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- `.pre-commit-config.yaml` with ruff and pre-commit-hooks
- `.editorconfig` for cross-editor formatting consistency
- `Makefile` with unified `make ci` command
- Dependabot configuration for pip, github-actions, and npm
- PR template with HISTORY/CHANGELOG checklist
- Issue templates for bug reports and feature requests

## [0.4.0] - 2026-03-21

### Added
- CodeGen export: `w.export_code()` generates standalone LightGBM training/prediction code
- Browser download for exported code via binary buffer (JupyterLab, VS Code, Colab)
- BlockedGroupKFold CV strategy with interactive 2-axis configuration UI
- Column stats inspection (`get_column_stats` action) for data-driven CV setup
- Fold preview visualization (`preview_splits` action) with period flow diagram
- Tune cancel support via unified `_run_with_cancel_polling` routing
- Release automation: `scripts/release.py` + `auto-release.yml` + `release.yml`
- CI distribution check with twine validation and smoke test
- CI coverage threshold (80% minimum)
- CHANGELOG.md with Keep a Changelog format

### Fixed
- `blocked_group_kfold` missing from `_FALLBACK_STRATEGIES` (silent CV_ERROR)
- `apply_loaded_config` dropping blocks/groups fields on config round-trip
- Temp file cleanup using separate `contextlib.suppress` blocks
- Cutoff comparison type mismatch in `preview_splits`
- `[dependency-groups]` dev not including ruff/mypy/pytest (CI failure)

### Changed
- Require lizyml >= 0.4.0
- Replace `publish.yml` (release-event trigger) with `release.yml` (tag trigger)
- Export Code sends zip as binary buffer instead of server-side path string

## [0.3.0] - 2026-03-21

### Added
- Google Colab compatibility: JS polling fallback for background thread comm blackout
- Dark mode support with CSS variable layer (`--lzw-*`) and theme toggle button
- WCAG AA contrast ratio CI tests (68 test cases)
- Plotly chart dark mode theme tracking
- CodeGen export: browser download via binary buffer (works on JupyterLab, VS Code, Colab)
- `w.export_code(path)` Python API for generating standalone training/prediction code
- BlockedGroupKFold CV strategy with interactive UI (distribution bars, fold preview)
- `get_column_stats` and `preview_splits` actions for data-driven CV configuration
- Tune cancel support via `_run_with_cancel_polling` unification

### Fixed
- OpenMP thread pool accumulation causing CPU degradation on repeated Fit/Tune (join previous worker thread)
- Colab polling limited to Colab-only via `isColab()` detection (prevents GIL contention on VS Code/JupyterLab)
- `--lzw-fg-2` contrast ratio below WCAG AA (#999 → #767676)
- Host environment CSS variable leakage (`--jp-*` override on Colab/VS Code)
- VS Code Notebook table/button white background in dark mode
- Dropdown menu visibility in dark mode
- Segment button active state visibility

### Changed
- Require lizyml >= 0.4.0
- `_handle_custom_msg` override signature: `(self, content, buffers)` per ipywidgets protocol
- Export Code sends zip as binary buffer instead of server-side path string

## [0.2.1] - 2026-03-14

### Fixed
- OpenMP daemon thread degradation workaround (daemon=False)
- Cancel-polling pattern for blocking library calls

## [0.2.0] - 2026-03-10

### Added
- Tune tab with search space configuration (Fixed/Range/Choice modes)
- Apply best params from Tune to Fit config
- Dynamic form generation from backend contract schema
- YAML import/export for config
- Inference tab with prediction table and SHAP support

## [0.1.0] - 2026-02-28

### Added
- Initial release
- LizyWidget with Data, Model (Fit), and Results tabs
- Auto-detection of task type, CV strategy, and column settings
- LightGBM backend via LizyMLAdapter
- Plotly chart integration (Learning Curve, ROC, Feature Importance, etc.)
- Python API: `load()`, `fit()`, `tune()`, `predict()`, `save_model()`
