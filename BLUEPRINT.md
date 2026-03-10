## 0. ステータスとスコープ

### スコープ内

- Jupyter Notebook / Google Colab / VS Code Notebooks 向け補助 UI Widget
- DataFrame を渡してコードなしで LizyML の Fit・Tune・Inference を実行できる
- Fit/Tune 結果（メトリクス・プロット・特徴量重要度）のインライン表示
- Config の編集・Import/Export（YAML）
- Inference（学習済みモデルで新データを予測）
- 複数 ML バックエンドの Adapter 対応（初期バックエンドは LizyML）

### スコープ外

- 単独サーバー起動・ブラウザ操作（LizyStudio の担当）
- マルチユーザー・認証・権限管理
- ファイルブラウザ / ファイルアップロード UI（ユーザーが Python でデータを用意するのが前提）
- ジョブのディスク永続化（セッション内メモリのみ。保存は明示的 API で行う）
- モバイル対応

---

## 1. 目的

```python
from lizyml_widget import LizyWidget

df = pd.read_csv("train.csv")
w = LizyWidget()
w.load(df, target="price")
w   # セルに置くだけで UI 表示
```

`pip install lizyml-widget` だけで使えるインライン Widget を提供する。
LizyML の公開 API をそのまま呼び出し、Widget 独自の ML ロジックは持たない。

---

## 2. 設計原則

| # | 原則 | 説明 |
|---|------|------|
| 1 | **バックエンドに忠実** | Widget 独自の ML ロジックを持たない。Adapter 経由でバックエンドライブラリの API をそのまま呼ぶ |
| 2 | **Python API ファースト** | UI 操作とプログラム操作の両立。`widget.load(df)` 等の Python API を提供する |
| 3 | **最小バンドル** | anywidget + Preact。Plotly は CDN 動的ロード。gzip 後のバンドルサイズ目標: JS 50KB 以下（Plotly 除く） |
| 4 | **環境非依存** | Jupyter Notebook / Google Colab / VS Code Notebooks のいずれでも動作する |
| 5 | **バックエンドの仕様が正** | Config schema・Result 型等は各バックエンドライブラリが正。Widget 側で再定義しない |
| 6 | **責務分離** | UI（Preact）← traitlets →　Python（Service → Adapter → LizyML）の2層 |

---

## 3. アーキテクチャ

### 3.1 全体構成

```
┌──────────────────────────────────────────┐
│          Jupyter / Colab / VS Code        │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │         LizyWidget (Cell output)   │  │
│  │                                    │  │
│  │  Preact (index.tsx)                │  │
│  │  ├─ DataTab                        │  │
│  │  ├─ ConfigTab (Fit / Tune)         │  │
│  │  └─ ResultsTab                     │  │
│  │        │ anywidget traitlets        │  │
│  └────────┼───────────────────────────┘  │
│           │ Python kernel                │
│  ┌────────┴───────────────────────────┐  │
│  │     LizyWidget (anywidget)         │  │
│  │     ├─ WidgetService               │  │
│  │     └─ BackendAdapter              │  │
│  │           │ Python import           │  │
│  │     ┌─────┴──────┐                 │  │
│  │     │   LizyML   │  (Future: ...)  │  │
│  │     └────────────┘                 │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### 3.2 レイヤー責務

| レイヤー | 場所 | 責務 | 禁止事項 |
|---------|------|------|---------|
| **UI** | `js/src/` | 表示・操作。traitlets を読み書きする | ML ロジック・Python 直接呼び出し |
| **Widget** | `src/lizyml_widget/widget.py` | traitlets 定義・Action 処理・スレッド管理 | ML ロジック直接記述 |
| **Service** | `src/lizyml_widget/service.py` | 状態管理・Adapter 呼び出し調整・自動判定ロジック | バックエンド固有の型の直接利用 |
| **Adapter** | `src/lizyml_widget/adapter.py` | バックエンドライブラリの呼び出し・共通型への変換 | Widget / traitlets の知識 |

### 3.3 Backend Adapter 設計

LizyStudio BLUEPRINT §3.3 と共通の設計を使用する。

#### 3.3.1 共通型

```python
# src/lizyml_widget/types.py

from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class BackendInfo:
    name: str       # "lizyml"
    version: str    # "0.x.x"

@dataclass
class ConfigSchema:
    json_schema: dict[str, Any]

@dataclass
class FitSummary:
    metrics: dict[str, Any]
    fold_count: int
    params: list[dict[str, Any]]

@dataclass
class TuningSummary:
    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]]
    metric_name: str
    direction: str   # "minimize" | "maximize"

@dataclass
class PredictionSummary:
    predictions: pd.DataFrame
    warnings: list[str]

@dataclass
class PlotData:
    plotly_json: str   # fig.to_json()
```

#### 3.3.2 BackendAdapter Protocol

```python
# src/lizyml_widget/adapter.py

from typing import Protocol, Any, Callable
import pandas as pd
from .types import BackendInfo, ConfigSchema, FitSummary, TuningSummary, PredictionSummary, PlotData

class BackendAdapter(Protocol):

    @property
    def info(self) -> BackendInfo: ...

    def get_config_schema(self) -> ConfigSchema: ...
    def validate_config(self, config: dict) -> list[dict]: ...

    def create_model(self, config: dict, dataframe: pd.DataFrame) -> Any: ...
    def fit(self, model: Any, *, params: dict | None = None, on_progress: Callable | None = None) -> FitSummary: ...
    def tune(self, model: Any, *, on_progress: Callable | None = None) -> TuningSummary: ...
    def predict(self, model: Any, data: pd.DataFrame, *, return_shap: bool = False) -> PredictionSummary: ...

    def evaluate_table(self, model: Any) -> list[dict]: ...
    def split_summary(self, model: Any) -> list[dict]: ...
    def importance(self, model: Any, kind: str) -> dict[str, float]: ...
    def plot(self, model: Any, plot_type: str) -> PlotData: ...
    def available_plots(self, model: Any) -> list[str]: ...

    def export_model(self, model: Any, path: str) -> str: ...
    def load_model(self, path: str) -> Any: ...
    def model_info(self, model: Any) -> dict[str, Any]: ...
```

### 3.4 State Sync（traitlets）

Python と JS の状態は anywidget の traitlets で同期する。

#### 方向の定義

- **Python → JS**: Python がセットし、JS は読み取り専用とする
- **JS → Python**: JS が `action` traitlet に書き込み、Python が `observe` で処理する

#### traitlets 一覧

| traitlet 名 | 型 | 方向 | 説明 |
|------------|-----|------|------|
| `backend_info` | `Dict` | P→JS | `{"name": "lizyml", "version": "0.x.x"}` |
| `df_info` | `Dict` | P→JS | DataFrame のメタ情報（§3.5 参照） |
| `config_schema` | `Dict` | P→JS | Config の JSON Schema |
| `config` | `Dict` | P→JS | 現在の Config（ユーザー操作で更新） |
| `status` | `Unicode` | P→JS | `"idle"` \| `"data_loaded"` \| `"running"` \| `"completed"` \| `"failed"` |
| `job_type` | `Unicode` | P→JS | `"fit"` \| `"tune"` \| `""` |
| `job_index` | `Int` | P→JS | セッション内のジョブ連番（1, 2, 3…） |
| `progress` | `Dict` | P→JS | `{"current": 2, "total": 5, "message": "Fold 2/5..."}` |
| `elapsed_sec` | `Float` | P→JS | 実行開始からの経過秒数（1秒ごと更新） |
| `fit_summary` | `Dict` | P→JS | FitSummary のシリアライズ結果 |
| `tune_summary` | `Dict` | P→JS | TuningSummary のシリアライズ結果 |
| `available_plots` | `List` | P→JS | 利用可能な plot_type 一覧 |
| `inference_result` | `Dict` | P→JS | 直近の Inference 結果サマリー |
| `error` | `Dict` | P→JS | `{"code": "BACKEND_ERROR", "message": "..."}` |
| `action` | `Dict` | JS→P | JS からの命令（§3.6 参照） |

#### 3.5 df_info の構造

```json
{
  "shape": [1000, 20],
  "target": "price",
  "task": "regression",
  "columns": [
    {
      "name": "age",
      "dtype": "int64",
      "unique_count": 50,
      "suggested_type": "numeric",
      "suggested_excluded": false,
      "exclude_reason": null,
      "excluded": false,
      "col_type": "numeric"
    }
  ],
  "cv": {
    "strategy": "kfold",
    "n_splits": 5,
    "group_column": null
  },
  "feature_summary": {
    "total": 15,
    "numeric": 10,
    "categorical": 5,
    "excluded": 3,
    "excluded_id": 1,
    "excluded_const": 1,
    "excluded_manual": 1
  }
}
```

`target` 選択時に Service 層が自動判定（§5.2 参照）を行い `df_info` を更新する。

### 3.6 Action パターン（JS → Python）

JS は `action` traitlet に Dict を書き込む。Python の `@observe("action")` が処理する。

| `type` | `payload` | 説明 |
|--------|-----------|------|
| `"fit"` | `{}` | 現在の config + df で fit を実行 |
| `"tune"` | `{}` | 現在の config + df で tune を実行 |
| `"cancel"` | `{}` | 実行中ジョブをキャンセル |
| `"set_target"` | `{"target": "price"}` | Target カラムを変更し df_info を再計算 |
| `"update_column"` | `{"name": "age", "excluded": false, "col_type": "numeric"}` | 特定カラムの設定を変更 |
| `"update_cv"` | `{"strategy": "kfold", "n_splits": 5, "group_column": null}` | CV 設定を変更 |
| `"update_config"` | `{"config": {...}}` | Config dict を更新 |
| `"request_plot"` | `{"plot_type": "roc-curve"}` | プロット JSON をリクエスト（メッセージで返す） |
| `"request_inference_plot"` | `{"plot_type": "roc-curve"}` | Inference プロットをリクエスト |
| `"run_inference"` | `{"return_shap": false}` | `inference_df` traitlet のデータで推論実行 |

プロット JSON は traitlet ではなくカスタムメッセージ（`widget.send()`）で返す。サイズが大きいため、すべてをメモリに保持しない。

```python
# Python 側
self.send({"type": "plot_data", "plot_type": "roc-curve", "plotly_json": "..."})
```

```ts
// JS 側
model.on("msg:custom", (msg) => {
  if (msg.type === "plot_data") renderPlot(msg.plot_type, msg.plotly_json);
});
```

### 3.7 非同期実行

fit / tune は `threading.Thread` でバックグラウンド実行する。

```python
def _run_fit(self):
    def on_progress(current, total, message):
        self.progress = {"current": current, "total": total, "message": message}
        self.elapsed_sec = (time.monotonic() - self._start_time)

    try:
        self.status = "running"
        result: FitSummary = self._service.fit(on_progress=on_progress)
        self.fit_summary = asdict(result)
        self.status = "completed"
    except Exception as e:
        self.error = {"code": "BACKEND_ERROR", "message": str(e)}
        self.status = "failed"
```

anywidget は traitlets への書き込みをスレッドセーフに処理するため、バックグラウンドスレッドからの更新が安全に JS へ伝播する。

---

## 4. Python API

### 4.1 LizyWidget クラス

```python
class LizyWidget(anywidget.AnyWidget):
    """LizyML の Notebook 補助 UI Widget。"""

    def load(
        self,
        df: pd.DataFrame,
        target: str | None = None,
    ) -> "LizyWidget":
        """DataFrame を読み込み Data タブを初期化する。"""

    def set_config(self, config: dict) -> "LizyWidget":
        """Config dict を直接セットする（YAML ロードとの連携用）。"""

    def load_config(self, path: str) -> "LizyWidget":
        """YAML / JSON ファイルから Config を読み込む。"""

    def save_config(self, path: str) -> None:
        """現在の Config を YAML ファイルに書き出す。"""

    def get_fit_summary(self) -> FitSummary | None:
        """直近の Fit 結果を Python オブジェクトで返す（コード連携用）。"""

    def get_tune_summary(self) -> TuningSummary | None:
        """直近の Tune 結果を Python オブジェクトで返す。"""

    def get_model(self) -> Any | None:
        """直近のモデルオブジェクトを返す（バックエンド依存型）。"""

    def predict(
        self,
        df: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionSummary:
        """UI を介さずにプログラムから推論を実行する。"""

    def save_model(self, path: str) -> str:
        """直近のモデルを指定パスに保存する。保存先パスを返す。"""
```

メソッドチェーンをサポートする（`load()` / `set_config()` は `self` を返す）。

```python
# 使用例
w = LizyWidget().load(df, target="price").load_config("lgbm.yaml")
w
```

---

## 5. UI 仕様

### 5.1 Widget 外観・レイアウト

セル出力にインライン表示される単一 Widget。横幅 100%（セル幅に追従）、高さ 480px（固定、内部スクロール）。

```
┌──────────────────────────────────────────────────┐
│ ⚡ LizyML Widget    lizyml v0.x.x    ● Idle      │  ← ヘッダー (40px)
├──────────────────────────────────────────────────┤
│ [▶ Data]  [ Config]  [ Results]                  │  ← タブバー (36px)
├──────────────────────────────────────────────────┤
│                                                  │
│  (タブコンテンツ、内部スクロール)                    │  ← コンテンツ (残り)
│                                                  │
└──────────────────────────────────────────────────┘
```

**ヘッダー:**

| 要素 | 説明 |
|------|------|
| アイコン + タイトル | `⚡ LizyML Widget` |
| バックエンドバッジ | `lizyml v0.x.x`（`backend_info` から表示） |
| ステータスバッジ | `● Idle` / `● Data Loaded` / `⟳ Running` / `✓ Completed` / `✗ Failed` |

**タブ切り替え:**

| タブ | 有効条件 |
|------|---------|
| Data | 常時 |
| Config | `status` が `data_loaded` 以降 |
| Results | `status` が `completed` または `failed` 以降（一度でも実行済み） |

完了後は Results タブに自動切り替え。

---

### 5.2 Data タブ

DataFrame のメタ情報表示・Target 選択・Column 設定・CV 設定を行う。

```
┌──────────────────────────────────────────────────┐
│ DataFrame: 1000 rows × 20 cols                   │
│                                                  │
│ ▸ Target / Task                                  │
│   Target  [price              ▼]                 │
│   Task    [regression  ▼] ⚡auto                 │
│                                                  │
│ ▸ Column Settings                                │
│   ┌─────────────┬──────┬──────┬────────────┐    │
│   │ Column      │ Uniq │ Excl │ Type       │    │
│   ├─────────────┼──────┼──────┼────────────┤    │
│   │ id          │ 1000 │ ☑    │ ── [ID]    │    │
│   │ const_col   │    1 │ ☑    │ ── [Const] │    │
│   │ age         │   50 │ ☐    │ Numeric ▼  │    │
│   │ city        │   15 │ ☐    │ Categ.  ▼  │    │
│   └─────────────┴──────┴──────┴────────────┘    │
│                                                  │
│ ▸ Cross Validation                               │
│   [KFold              ▼]  [5] folds              │
│                                                  │
│ ── Features: 15 cols (Numeric: 10, Categ.: 5) ── │
│ ── Excluded: 3 (ID: 1, Const: 1, Manual: 1)  ── │
└──────────────────────────────────────────────────┘
```

#### Target / Task

| 要素 | 動作 |
|------|------|
| Target | `df_info.columns` 全カラムのドロップダウン。選択時に Task 自動判定と Column 自動設定を実行 |
| Task | 自動判定結果を初期値とするドロップダウン（KFold / StratifiedKFold / regression）。変更可能 |

**Task 自動判定ルール（Service 層が実装）:**

| 目的変数の条件 | 判定 |
|--------------|------|
| ユニーク数 = 2 | `binary` |
| dtype が object / category かつユニーク数 > 2 | `multiclass` |
| dtype が数値 かつユニーク数 ≤ `max(20, 行数 × 0.05)` | `multiclass` |
| 上記以外 | `regression` |

#### Column Settings

Target 選択時に一括実行される自動設定:

| 条件 | 設定 |
|------|------|
| ユニーク数 = 行数 | `excluded: true` + `exclude_reason: "id"` |
| ユニーク数 = 1 | `excluded: true` + `exclude_reason: "constant"` |
| dtype が object / string / category / bool | `col_type: "categorical"` |
| dtype が数値 かつユニーク数 ≤ `max(20, 行数 × 0.05)` | `col_type: "categorical"` |
| 上記以外 | `col_type: "numeric"` |

ユーザーは自動設定後にチェックボックスとドロップダウンで各カラムを変更できる。
`excluded: true` の行は Type ドロップダウンをグレーアウト表示する。

#### Cross Validation

| 要素 | 説明 |
|------|------|
| Strategy | KFold / StratifiedKFold / GroupKFold / TimeSeriesSplit |
| Folds | 数値入力（デフォルト: 5） |
| Group column | GroupKFold 選択時のみ表示。特徴量カラムのドロップダウン |

Task に応じたデフォルト: `binary` / `multiclass` → `StratifiedKFold`、`regression` → `KFold`。

#### Data Panel → Config 自動反映

| Data タブの設定 | Config フィールド |
|----------------|-----------------|
| Target | `data.target` |
| Task | `data.task` |
| Type = Categorical かつ Excl OFF | `features.categorical` |
| Excl = ON | `features.exclude` |
| CV Strategy | `split.strategy` |
| CV Folds | `split.n_splits` |
| Group column | `split.group_column` |

---

### 5.3 Config タブ

Fit サブタブと Tune サブタブで構成する。

```
┌──────────────────────────────────────────────────┐
│ [▶ Fit]  [ Tune]                  [━━ Fit ━━]   │  ← sticky
├──────────────────────────────────────────────────┤
│ ▸ Model                                          │
│   [LightGBM                      ▼]              │
│   learning_rate        [0.1       ]              │
│   num_leaves           [31        ]              │
│   n_estimators         [1000      ]              │
│   max_depth            [-1        ]              │
│                                                  │
│ ▸ Training                                       │
│   early_stopping_rounds  [50      ]              │
│   verbose                [100     ]              │
│                                                  │
│ ▸ Evaluation                                     │
│   ☑ AUC  ☑ LogLoss  ☐ Accuracy  ☐ F1           │
│                                                  │
│ ▸ Calibration ──────────────────── [ON | OFF]   │  ← binary のみ表示
│   method [isotonic          ▼]                   │
│                                                  │
│ [Import YAML]  [Export YAML]  [Raw Config]       │
└──────────────────────────────────────────────────┘
```

#### フォーム動的生成

フォームフィールドは `config_schema`（JSON Schema）から動的に生成する。

| JSON Schema 型 | フォームコンポーネント |
|----------------|---------------------|
| `number` / `integer` | NumberInput（+/- ステッパー付き） |
| `boolean` | Switch / Toggle |
| `string` + `enum` | Select ドロップダウン |
| `string` | TextInput |
| `array` | タグ形式のマルチ入力 |

- `default` → フォーム初期値
- `description` → ツールチップ表示
- フォーム変更時に `update_config` action でデバウンスして Python に送信
- Python 側がバリデーションし結果を `config` traitlet に反映

#### Tune サブタブ

Fit タブと独立した Search Space 設定。

```
┌──────────────────────────────────────────────────┐
│ [ Fit]  [▶ Tune]                  [━━ Tune ━━]  │
├──────────────────────────────────────────────────┤
│ ▸ Model                                          │
│   [LightGBM                      ▼]              │
│                                                  │
│ ▸ Settings                                       │
│   n_trials      [100    ▼]                       │
│   timeout (sec) [600    ▼]                       │
│   scoring       [AUC    ▼]                       │
│                                                  │
│ ▸ Search Space                                   │
│   ┌─────────────┬──────────┬──────────────────┐  │
│   │ Param       │ Mode     │ Config           │  │
│   ├─────────────┼──────────┼──────────────────┤  │
│   │ ▸ lr        │ [Range▼] │ 0.01 ~ 0.3 (Log) │  │
│   │ ▸ n_leaves  │ [Range▼] │ 10 ~ 100         │  │
│   │ ▸ n_estim   │ [Fixed▼] │ 1000             │  │
│   └─────────────┴──────────┴──────────────────┘  │
│                                                  │
│ [Import YAML]  [Export YAML]  [Raw Config]       │
└──────────────────────────────────────────────────┘
```

Search Space の Mode 選択肢はパラメータ型で決まる:

| パラメータ型 | 選択可能な Mode |
|------------|---------------|
| `float` / `integer` | Fixed / Range |
| `enum` / `string` | Fixed / Choice |
| `boolean` | Fixed / Choice |

**Fit / Tune ボタン有効条件:**

| 条件 | Fit | Tune |
|------|-----|------|
| `status` が `data_loaded` 以降 | 必須 | 必須 |
| Model 選択済み | 必須 | 必須 |
| Range/Choice パラメータが 1 つ以上 | 不要 | 必須 |

#### Config Import / Export

| 操作 | 説明 |
|------|------|
| Import YAML | YAML / JSON ファイルを Widget 内のファイルピッカーで選択して読み込む。`data` / `features` / `split` は Data タブに反映、その他は Config フォームに反映 |
| Export YAML | 現在の全設定（Data + Config 両タブ）を YAML でダウンロード |
| Raw Config | フル Config の YAML テキストビュー（読み取り専用モーダル） |

---

### 5.4 Results タブ

#### 実行中

```
┌──────────────────────────────────────────────────┐
│ Fit #3 ── LightGBM  ⟳ Running                    │
│                                                  │
│ ████████████░░░░░░  Fold 3 / 5                   │
│ Elapsed: 00:42                                   │
│                                                  │
│ Fold 1  AUC = 0.889  ✓                          │
│ Fold 2  AUC = 0.901  ✓                          │
│ Fold 3  training...                              │
│ Fold 4  ─                                        │
│ Fold 5  ─                                        │
│                                                  │
│                                  [Cancel]        │
└──────────────────────────────────────────────────┘
```

Tune 実行中は Fold の代わりに Trial 進捗を表示（`Trial 12 / 100  Best AUC = 0.891`）。

#### Fit 完了

タブ分割せず1つのスクロールビューで表示する。

```
┌──────────────────────────────────────────────────┐
│ Fit #3 ── LightGBM  ✓ Completed                  │
│                                                  │
│ ── Score ──                                      │
│ ┌──────┬────────┬────────┬──────────┐            │
│ │      │   IS   │   OOS  │ OOS Std  │            │
│ ├──────┼────────┼────────┼──────────┤            │
│ │ AUC  │ 0.952  │ 0.892  │  0.012   │            │
│ │ LogL │ 0.198  │ 0.341  │  0.008   │            │
│ └──────┴────────┴────────┴──────────┘            │
│                                                  │
│ ── Learning Curve ──                             │
│ ┌──────────────────────────────────────────────┐ │
│ │            (Plotly chart)                    │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ ── Plots  [ROC Curve              ▼] ──          │
│ ┌──────────────────────────────────────────────┐ │
│ │            (Plotly chart)                    │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ ▸ Feature Importance                             │
│ ▸ Fold Details        ← CV 時のみ               │
│ ▸ Parameters                                     │
└──────────────────────────────────────────────────┘
```

**Score テーブル:**

| 列 | 説明 |
|----|------|
| IS | In Sample スコア |
| OOS | Out of Sample スコア（CV 時は OOF 値） |
| OOS Std | Fold 間標準偏差。**CV 時のみ表示**（非 CV 時は列を非表示） |

**Plots セレクタ:**

| plot_type | 表示条件 |
|-----------|---------|
| `learning-curve` | 全タスク（常時） |
| `oof-distribution` | 全タスク |
| `residuals` | regression |
| `roc-curve` | binary |
| `calibration` | binary + calibration 有効 |
| `probability-histogram` | binary |

セレクタ変更時に `request_plot` action を送信。Python がプロット生成し `msg:custom` で返す。

**Accordion セクション:**

| セクション | 内容 | 表示条件 |
|-----------|------|---------|
| Feature Importance | 特徴量重要度の棒グラフ（Plotly） | 常時 |
| Fold Details | Fold 別メトリクスとデータサイズのテーブル | CV 時のみ |
| Parameters | 使用ハイパーパラメータ一覧 | 常時 |

#### Tune 完了

Fit 完了と同じ構成に加え、先頭に探索結果を追加する。

```
Optimization History → Best Params → [Apply to Fit ▸]
→ Score → Learning Curve → Plots → Accordion(Trial Results / Feature Importance / Fold Details / Parameters)
```

**Optimization History:** Trial 番号 vs スコアの収束プロット（Plotly）。

**Best Params + Apply to Fit:** Best Params テーブルと、Fit サブタブへコピーして切り替えるボタン。

#### Inference セクション（Results タブ内）

Fit / Tune 完了後、Results タブの末尾に Inference セクションを追加表示する。

```
┌──────────────────────────────────────────────────┐
│ ── Inference ──                                  │
│                                                  │
│ DataFrame  [widget.inference_df が未セット時]     │
│ ⚠ Call widget.load_inference(df) to set data.    │
│                                                  │
│ [☐ Return SHAP values]                           │
│                                                  │
│ [Run Inference]  ← inference_df セット後に有効    │
└──────────────────────────────────────────────────┘
```

推論データは `widget.load_inference(df)` で Python からセットする（ブラウザアップロードなし）。

推論完了後の結果表示:

```
── Inference Result ──
┌────────┬──────┬────────┐
│ idx    │ pred │ proba  │
├────────┼──────┼────────┤
│ 0      │  1   │ 0.87   │
│ 1      │  0   │ 0.23   │
└────────┴──────┴────────┘
Showing 50 of 500 rows    [Download CSV]

▸ Prediction Distribution
▸ SHAP Summary        ← SHAP 有効時のみ
▸ Warnings            ← 警告がある場合のみ
```

#### エラー表示

```
┌──────────────────────────────────────────────────┐
│ Fit #3 ── LightGBM  ✗ Failed                     │
│                                                  │
│ BACKEND_ERROR                                    │
│ Feature 'age' has unsupported dtype 'object'     │
│                                                  │
│ [Show Full Traceback]        [Re-run]            │
└──────────────────────────────────────────────────┘
```

---

## 6. エラーハンドリング

### 6.1 エラーコード

| コード | 意味 |
|--------|------|
| `NO_DATA` | `load()` が呼ばれていない |
| `NO_TARGET` | Target が未選択 |
| `VALIDATION_ERROR` | Config バリデーション失敗 |
| `BACKEND_ERROR` | バックエンドライブラリの内部エラー |
| `CANCELLED` | ユーザーによるキャンセル |
| `INTERNAL_ERROR` | 予期しないエラー |

### 6.2 エラー伝播

バックエンドライブラリのエラーは `BACKEND_ERROR` として `error` traitlet に格納する。`message` に元のエラーメッセージ、`traceback` に完全なトレースバックを含める。

```python
self.error = {
    "code": "BACKEND_ERROR",
    "message": str(e),
    "traceback": traceback.format_exc(),
}
```

---

## 7. フロントエンド設計

### 7.1 技術スタック

| 要素 | 採用 | 理由 |
|------|------|------|
| Widget フレームワーク | **anywidget** | Jupyter / Colab / VS Code 対応。`_esm` + traitlets のみで動作 |
| UI ライブラリ | **Preact + JSX** | gzip 3KB。React 互換 API で BLUEPRINT 知識を再利用 |
| ビルド | **esbuild** | 設定ゼロに近い。`--bundle --format=esm` の1コマンドで完結 |
| CSS | **スコープ付き Plain CSS** | `.lzw-` プレフィックスでグローバル汚染を防止 |
| グラフ | **Plotly.js（CDN 動的ロード）** | バンドルに含めず動的 `import()` で取得。Colab はインターネット接続前提 |

### 7.2 コンポーネント構成

```
js/src/
├── index.tsx          # anywidget ESM エントリ。モデル受け取りと Preact mount
├── App.tsx            # ヘッダー + タブルーター
├── tabs/
│   ├── DataTab.tsx
│   ├── ConfigTab.tsx
│   └── ResultsTab.tsx
├── components/
│   ├── Header.tsx         # ステータスバッジ + バックエンド情報
│   ├── ColumnTable.tsx    # Column Settings テーブル
│   ├── DynForm.tsx        # JSON Schema → フォーム動的生成
│   ├── SearchSpace.tsx    # Tune の Search Space 設定
│   ├── ScoreTable.tsx     # IS / OOS / OOS Std テーブル
│   ├── PlotViewer.tsx     # Plotly プロットセレクタ + 表示
│   ├── ProgressView.tsx   # プログレスバー + Fold ログ
│   ├── ParamsTable.tsx    # パラメータ一覧表
│   ├── PredTable.tsx      # Inference 結果テーブル（ページネーション付き）
│   └── Accordion.tsx      # 展開/折りたたみ
├── hooks/
│   ├── useModel.ts    # anywidget モデルの React/Preact 接続フック
│   └── usePlot.ts     # request_plot action + msg:custom 受信
└── widget.css         # .lzw- プレフィックス付きスタイル
```

### 7.3 anywidget エントリ

```ts
// js/src/index.tsx
import { render } from "preact";
import { App } from "./App";

function render_widget({ model, el }: { model: any; el: HTMLElement }) {
  el.classList.add("lzw-root");
  render(<App model={model} />, el);
  return () => render(null, el);  // cleanup
}

export default { render: render_widget };
```

### 7.4 Plotly 連携

Python 側が `fig.to_json()` を生成し、JS 側がカスタムメッセージで受け取って描画する。

```ts
// hooks/usePlot.ts
export function usePlot(model: any) {
  const [plots, setPlots] = useState<Record<string, any>>({});

  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === "plot_data") {
        setPlots(prev => ({ ...prev, [msg.plot_type]: JSON.parse(msg.plotly_json) }));
      }
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model]);

  const requestPlot = (plot_type: string) => {
    if (plots[plot_type]) return;  // キャッシュがあればリクエスト不要
    model.set("action", { type: "request_plot", payload: { plot_type } });
    model.save_changes();
  };

  return { plots, requestPlot };
}
```

```ts
// components/PlotViewer.tsx
async function renderPlot(el: HTMLElement, spec: any) {
  // window.Plotly があれば（Colab 等で注入済み）再利用
  const Plotly = (window as any).Plotly
    ?? await import("https://esm.sh/plotly.js-dist-min@latest");
  Plotly.newPlot(el, spec.data, spec.layout, { responsive: true });
}
```

### 7.5 CSS スコーピング

すべてのスタイルを `.lzw-` プレフィックスで限定する。

```css
/* widget.css */
.lzw-root {
  font-family: var(--jp-ui-font-family, -apple-system, sans-serif);
  font-size: 13px;
  box-sizing: border-box;
}
.lzw-header { ... }
.lzw-tabs { ... }
.lzw-tab-content { ... }
```

anywidget の `_css` に読み込まれ `<style>` タグとして挿入される。Shadow DOM は使用しない（Plotly 等の外部ライブラリとの相性のため）。

---

## 8. ビルドと配布

### 8.1 開発時

```bash
# ターミナル 1: JS watch ビルド
cd js && npx esbuild src/index.tsx \
  --bundle --format=esm --jsx-factory=h --jsx-fragment=Fragment \
  --outfile=../src/lizyml_widget/static/widget.js \
  --watch

# ターミナル 2: Python 環境
uv run jupyter lab    # anywidget が static/ の変更を自動検知して HMR
```

anywidget は `importlib.resources` でバンドルを読み込む。開発時は `_esm = Path("static/widget.js")` の形式でファイルパスを指定することで HMR が動作する。

### 8.2 プロダクションビルド

```bash
# JS ビルド（minify + tree-shaking）
cd js && npx esbuild src/index.tsx \
  --bundle --format=esm --minify \
  --jsx-factory=h --jsx-fragment=Fragment \
  --outfile=../src/lizyml_widget/static/widget.js

# Python パッケージビルド
uv build    # → dist/ に wheel を生成
```

`pyproject.toml` で `hatchling` の `artifacts` に `src/lizyml_widget/static/**` を指定し、ビルド済み JS / CSS を wheel に含める。

### 8.3 Widget 読み込み（Python）

```python
# src/lizyml_widget/widget.py
import anywidget
import importlib.resources

class LizyWidget(anywidget.AnyWidget):
    _esm = importlib.resources.files("lizyml_widget") / "static/widget.js"
    _css = importlib.resources.files("lizyml_widget") / "static/widget.css"
```

### 8.4 PyPI 配布

- 単一パッケージ `lizyml-widget` として配布
- ビルド済み JS を内包（ユーザーに Node.js 不要）
- `pip install lizyml-widget` のみで利用可能

---

## 9. テスト戦略

### 9.1 Python（pytest）

| レベル | 対象 | 方針 |
|--------|------|------|
| Unit | Service 層の自動判定ロジック（Task 判定・Column 設定） | モック DataFrame で境界値テスト |
| Unit | Adapter の型変換（FitSummary, TuningSummary の構造） | LizyML の戻り値をモックで確認 |
| Integration | `LizyWidget.load()` → `fit()` → `fit_summary` traitlet | 小サイズの実データで E2E |

```bash
uv run pytest
uv run ruff check .
uv run mypy src/lizyml_widget/
```

### 9.2 フロントエンド

初期段階では自動テストを設けない。フォーム動的生成・traitlets 連携は Python 統合テストでカバーし、UI は手動確認とする。
必要に応じて HISTORY.md で Proposal の上で Vitest を導入する。

---

## 10. 開発コマンド

```bash
# JS
cd js
pnpm install
pnpm build          # 本番ビルド
pnpm dev            # watch モード

# Python
uv run jupyter lab                 # 開発（HMR 有効）
uv run pytest                      # テスト
uv run ruff check .                # Lint
uv run ruff format --check .       # Format check
uv run mypy src/lizyml_widget/     # 型チェック
```

---

## 11. ディレクトリ構成

```
lizyml-widget/
├── pyproject.toml
├── uv.lock
├── BLUEPRINT.md
├── CLAUDE.md
├── HISTORY.md
├── PLAN.md
├── js/
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── tsconfig.json
│   └── src/
│       ├── index.tsx
│       ├── App.tsx
│       ├── widget.css
│       ├── tabs/
│       │   ├── DataTab.tsx
│       │   ├── ConfigTab.tsx
│       │   └── ResultsTab.tsx
│       ├── components/
│       │   ├── Header.tsx
│       │   ├── ColumnTable.tsx
│       │   ├── DynForm.tsx
│       │   ├── SearchSpace.tsx
│       │   ├── ScoreTable.tsx
│       │   ├── PlotViewer.tsx
│       │   ├── ProgressView.tsx
│       │   ├── ParamsTable.tsx
│       │   ├── PredTable.tsx
│       │   └── Accordion.tsx
│       └── hooks/
│           ├── useModel.ts
│           └── usePlot.ts
├── src/lizyml_widget/
│   ├── __init__.py
│   ├── widget.py          # LizyWidget (anywidget)
│   ├── service.py         # WidgetService（状態管理・Adapter 呼び出し）
│   ├── adapter.py         # BackendAdapter Protocol + LizyMLAdapter
│   ├── types.py           # 共通型（FitSummary 等）
│   └── static/            # esbuild 出力（パッケージに内包）
│       ├── widget.js
│       └── widget.css
└── tests/
    ├── test_service.py
    └── test_adapter.py
```
