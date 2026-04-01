# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
