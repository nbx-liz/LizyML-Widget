# LizyML Widget

Interactive Jupyter widget for [LizyML](https://github.com/lizyml/lizyml) — fit, tune, and run inference on machine learning models without writing code.

## Features

- **Data Tab** — Load a DataFrame, select target, configure columns and cross-validation
- **Model Tab** — Edit LightGBM hyperparameters, configure tuning search space
- **Results Tab** — View scores, Plotly plots, feature importance, and inference results
- **Config Import/Export** — Save and load configurations as YAML
- **Python API** — Programmatic access to all widget functionality

## Installation

```bash
pip install lizyml-widget
```

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

## Tutorials

| Notebook | Task | Dataset |
|----------|------|---------|
| [Quick Start](notebooks/tutorial.ipynb) | Binary (synthetic) | Synthetic data |
| [Regression](notebooks/tutorial_regression.ipynb) | Regression | California Housing (sklearn) |
| [Binary Classification](notebooks/tutorial_binary.ipynb) | Binary | Breast Cancer Wisconsin (sklearn) |
| [Multiclass Classification](notebooks/tutorial_multiclass.ipynb) | Multiclass | Wine (sklearn) |

## Supported Environments

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code Notebooks

## Development

```bash
# Python
uv sync --all-extras
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
