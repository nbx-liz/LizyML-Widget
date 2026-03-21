# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
