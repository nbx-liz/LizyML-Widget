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

## License

MIT
