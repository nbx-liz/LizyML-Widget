# LizyML Widget

[![PyPI](https://img.shields.io/pypi/v/lizyml-widget)](https://pypi.org/project/lizyml-widget/)
[![Python](https://img.shields.io/pypi/pyversions/lizyml-widget)](https://pypi.org/project/lizyml-widget/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Interactive Jupyter widget for [LizyML](https://github.com/lizyml/lizyml) — fit, tune, and run inference on machine learning models without writing code.

## Features

- **Data Tab** — Load a DataFrame, select target, configure columns and cross-validation
- **Config Tab** — Edit LightGBM hyperparameters, configure tuning search space
- **Results Tab** — View scores, Plotly plots, feature importance, and inference results
- **Config Import/Export** — Save and load configurations as YAML
- **Python API** — Programmatic access to all widget functionality

## Requirements

- Python `>= 3.10`
- Jupyter Notebook, JupyterLab, Google Colab, or VS Code Notebooks
- `lizyml >= 0.9.0, < 0.10` (auto-resolved by the `[lizyml]` extras)

## Installation

```bash
# Recommended: installs a compatible lizyml automatically
pip install "lizyml-widget[lizyml]"
```

The `[lizyml]` extras pin `lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10`
so `pip` / `uv` / `poetry` always select a backend version that matches the
widget's expected contract.

If you manage `lizyml` separately, install the widget without extras and pin
`lizyml` yourself:

```bash
pip install lizyml-widget
pip install "lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10"
```

> Widget's `LizyMLAdapter` validates the installed `lizyml` version at import
> time and raises a clear `ImportError` if the backend is out of range.

### Version compatibility

LizyML Widget is tightly coupled to the `lizyml` ML contract (types,
`BackendAdapter` protocol, tune/fit result shapes), so each minor version of
the widget targets a specific `lizyml` range:

| lizyml-widget | lizyml           | Highlights                                                    |
| ------------- | ---------------- | ------------------------------------------------------------- |
| **0.8.x**     | `>=0.9.0,<0.10`  | Re-tune (round progress, boundary expansion, tuning history, `w.retune()` API) |
| 0.7.x         | `>=0.7.0,<0.9`   | Calibration / Search Space default refresh                    |
| 0.6.x / 0.5.x | `>=0.5.0,<0.7`   | Learning curve metric filter, CV strategy metadata            |

See [docs/VERSION_COMPAT.md](docs/VERSION_COMPAT.md) for the full matrix,
troubleshooting, and upgrade guidance.

## Quick Start

```python
import pandas as pd
from lizyml_widget import LizyWidget

df = pd.read_csv("train.csv")
w = LizyWidget()
w.load(df, target="price")
w  # display widget in notebook cell
```

### Programmatic Usage

```python
w = LizyWidget()
w.load(df, target="y").fit()

summary = w.get_fit_summary()
print(summary.metrics)

w.save_model("./model")
w.save_config("config.yaml")
```

### Re-tune (Study Resume + Boundary Expansion)

When the initial Tune run hits a search-space boundary or you simply want
more trials, call `w.retune()` to resume the existing Optuna study in
place.  The widget reuses the backend model so no history is lost.

```python
w = LizyWidget()
w.load(df, target="y")

# 1. Initial tune (e.g. 50 trials)
w.tune()

# 2. Resume with 30 more trials and let the backend widen boundaries
#    if the best trial lands on an edge.
w.retune(n_trials=30, expand_boundary=True, boundary_threshold=0.05)

summary = w.get_tune_summary()
for r in summary.rounds:
    print(f"Round {r['round']}: {r['n_trials']} trials, "
          f"best={r['best_score_after']}, expanded={r['expanded_dims']}")
```

The Results tab shows a **Re-tune (resume)** button inside the *Best Params*
accordion after the initial Tune completes — clicking it runs the same
resume flow from the UI, and the round-aware progress bar, Score History
chart, and Boundary Expansion panel update in place.

Requires `lizyml >= 0.9.0` (auto-resolved by `lizyml-widget[lizyml]`).

### Version

```python
import lizyml_widget
print(lizyml_widget.__version__)
```

## Tutorials

| Notebook | Task | Dataset |
|----------|------|---------|
| [Regression](notebooks/tutorial_regression.ipynb) | Regression | California Housing (sklearn) |
| [Binary Classification](notebooks/tutorial_binary.ipynb) | Binary | Breast Cancer Wisconsin (sklearn) |
| [Multiclass Classification](notebooks/tutorial_multiclass.ipynb) | Multiclass | Wine (sklearn) |

## Supported Environments

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code Notebooks

Powered by [anywidget](https://anywidget.dev/) for cross-environment compatibility.

## Development

```bash
# Python
uv sync --all-extras    # installs dev + lizyml dependencies
uv run pytest
uv run ruff check .
uv run mypy src/lizyml_widget/

# TypeScript
cd js
pnpm install
pnpm dev    # watch build
pnpm build  # production build
pnpm lint
```

### Stable Notebook Launch

If VS Code gets stuck reconnecting to an old kernel, prefer launching Jupyter with
workspace-local runtime files instead of the default global runtime directory:

```bash
./scripts/jupyter-reset.sh
./scripts/jupyter-lab.sh
```

This keeps runtime/config state under the repository and makes stale kernel/server
state easier to clear than relying on `Reload Window` alone.

## License

MIT
