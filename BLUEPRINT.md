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
│  │  ├─ ModelTab (ConfigTab: Fit / Tune) │  │
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
| **UI** | `js/src/` | `backend_contract` / `config` / `df_info` を描画し、ユーザー操作を `action` に変換する。ローカル状態は表示都合（例: Search Space の Mode）に限定する | ML ロジック・Python 直接呼び出し・backend 固有 option set / parameter catalog / step 値のハードコード・full config dict の合成 |
| **Widget** | `src/lizyml_widget/widget.py` | traitlets 定義・Action 処理・スレッド管理・`msg:custom` 中継 | ML ロジック直接記述、Service の private 状態参照、backend 固有 config 意味論の保持 |
| **Service** | `src/lizyml_widget/service.py` | Data タブ由来 state（target / task / columns / CV）の管理、実行前提判定、Adapter 呼び出し調整、canonical config と Data 系 state の結合 | バックエンド固有の default / option set / search space catalog / step 定数の保持 |
| **Adapter** | `src/lizyml_widget/adapter.py` | Backend Contract 提供、backend 固有 config default / patch 適用 / 実行前準備、バックエンドライブラリ呼び出し、共通型への変換 | Widget / traitlets の知識 |

`LizyWidget` は backend 差し替え・テスト容易性のため `adapter` をコンストラクタ引数で受け取れる。未指定時のみ既定の `LizyMLAdapter` を使用する。

完全疎結合化の基準は、backend 固有の UI/Config 知識が Adapter の返す **Backend Contract** にのみ存在することとする。UI と Service は contract を解釈して描画・調停するが、backend 依存の候補値や default を再定義しない。

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
class BackendContract:
    schema_version: int
    config_schema: dict[str, Any]
    ui_schema: dict[str, Any]
    capabilities: dict[str, Any]

@dataclass
class ConfigPatchOp:
    op: str              # "set" | "unset" | "merge"
    path: str            # dot path
    value: Any | None = None

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

`BackendContract.ui_schema` は JSON Schema にない UI 専用情報を保持する。初期設計では少なくとも以下を含む。

- セクション構成と表示順（Fit / Tune / Accordion / Search Space）
- field ごとの widget 種別、readOnly / visibility / label / help
- task や capability に応じた option set
- Search Space の parameter catalog、許可 mode、log scale 可否
- `lzw-stepper` 等の widget hint（`step`, `width`, `variant`）

#### 3.3.2 BackendAdapter Protocol

```python
# src/lizyml_widget/adapter.py

from typing import Protocol, Any, Callable, Literal, Sequence
import pandas as pd
from .types import (
    BackendInfo,
    BackendContract,
    ConfigPatchOp,
    FitSummary,
    TuningSummary,
    PredictionSummary,
    PlotData,
)

class BackendAdapter(Protocol):

    @property
    def info(self) -> BackendInfo: ...

    def get_backend_contract(self) -> BackendContract: ...
    def initialize_config(self, *, task: str | None) -> dict[str, Any]: ...
    def apply_config_patch(
        self,
        config: dict[str, Any],
        ops: Sequence[ConfigPatchOp],
        *,
        task: str | None,
    ) -> dict[str, Any]: ...
    def prepare_run_config(
        self,
        config: dict[str, Any],
        *,
        job_type: Literal["fit", "tune"],
        task: str | None,
    ) -> dict[str, Any]: ...
    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]: ...

    def create_model(self, config: dict[str, Any], dataframe: pd.DataFrame) -> Any: ...
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

`initialize_config()` / `apply_config_patch()` / `prepare_run_config()` により、backend 固有 default・可視性ルール・Tune 補完・legacy 正規化を Adapter へ一元化する。Service は Data タブ由来 state の保持と結合作業に専念し、UI は `BackendContract` の記述を generic に描画する。

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
| `backend_contract` | `Dict` | P→JS | Adapter 提供の contract（`config_schema` + `ui_schema` + `capabilities`） |
| `config` | `Dict` | P→JS | Python 側で正規化済みの canonical config |
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
| `action` | `Dict` | JS→P | JS からの命令（§3.6 参照）。config 編集は `patch_config` のみ |

#### 3.5 df_info の構造

```json
{
  "shape": [1000, 20],
  "target": "price",
  "task": "regression",
  "auto_task": "regression",
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
`auto_task` は自動判定結果を保持し、`task == auto_task` のとき UI は `⚡auto` を表示する。

### 3.6 Action パターン（JS → Python）

JS は `action` traitlet に Dict を書き込む。Python の `@observe("action")` が処理する。

| `type` | `payload` | 説明 |
|--------|-----------|------|
| `"fit"` | `{}` | 現在の config + df で fit を実行 |
| `"tune"` | `{}` | 現在の config + df で tune を実行 |
| `"cancel"` | `{}` | 実行中ジョブをキャンセル |
| `"set_target"` | `{"target": "price"}` | Target カラムを変更し df_info を再計算 |
| `"set_task"` | `{"task": "binary"}` | Task を手動変更し、Task 依存の Data state / canonical config を再計算 |
| `"update_column"` | `{"name": "age", "excluded": false, "col_type": "numeric"}` | 特定カラムの設定を変更 |
| `"update_cv"` | `{"strategy": "kfold", "n_splits": 5, "group_column": null}` | CV 設定を変更 |
| `"patch_config"` | `{"ops": [{"op": "set", "path": "training.seed", "value": 123}]}` | backend 固有 config への patch を適用し canonical config を再計算 |
| `"request_plot"` | `{"plot_type": "roc-curve"}` | プロット JSON をリクエスト（メッセージで返す） |
| `"request_inference_plot"` | `{"plot_type": "roc-curve"}` | Inference プロットをリクエスト |
| `"run_inference"` | `{"return_shap": false}` | `inference_df` traitlet のデータで推論実行 |
| `"import_yaml"` | `{"content": "yaml_string"}` | YAML 文字列を読み込み Config に反映 |
| `"export_yaml"` | `{}` | 現在の Config を YAML でダウンロード（msg:custom で返す） |
| `"raw_config"` | `{}` | フル Config の YAML テキストを取得（msg:custom で返す） |
| `"apply_best_params"` | `{"params": {...}}` | Tune 実行時 config を復元した上で Best Params を適用し、Fit 画面へ反映 |

Config 編集導線で JS が full config dict を送ることは禁止する。UI は表示用のローカル state（例: Search Space の `mode=Fixed/Range/Choice`）を保持できるが、Python へ送る payload は `ConfigPatchOp` に正規化したものだけとする。

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

    def __init__(self, *, adapter: BackendAdapter | None = None, **kwargs) -> None:
        """Widget を初期化する。`adapter` 未指定時は LizyMLAdapter を使用する。"""

    # ── 読み取り専用プロパティ ─────────────────────────────
    @property
    def task(self) -> str | None:
        """自動判定されたタスク種別（"binary" / "multiclass" / "regression"）。"""

    @property
    def cv_method(self) -> str:
        """現在の CV ストラテジー名（例: "stratified_kfold"）。"""

    @property
    def cv_n_splits(self) -> int:
        """CV の分割数。"""

    @property
    def df_shape(self) -> list[int]:
        """読み込み済み DataFrame の形状 [rows, cols]。"""

    @property
    def df_columns(self) -> list[dict]:
        """読み込み済み DataFrame のカラムメタデータ一覧。"""

    # ── データ操作 ─────────────────────────────────────────
    def load(
        self,
        df: pd.DataFrame,
        target: str | None = None,
    ) -> "LizyWidget":
        """DataFrame を読み込み Data タブを初期化する。"""

    def set_target(self, col: str) -> "LizyWidget":
        """ターゲットカラムを設定し自動判定を再実行する。"""

    def set_config(self, config: dict) -> "LizyWidget":
        """Config dict を受け取り canonicalization 後の snapshot を保持する。"""

    def load_config(self, path: str) -> "LizyWidget":
        """YAML / JSON ファイルから Config を読み込み canonicalization する。"""

    def save_config(self, path: str) -> None:
        """現在の Config を YAML ファイルに書き出す。"""

    # ── 実行 ───────────────────────────────────────────────
    def fit(self, *, timeout: float | None = None) -> "LizyWidget":
        """バックグラウンドスレッドで Fit を実行し完了まで待機する。
        失敗時は RuntimeError を raise する。"""

    def tune(self, *, timeout: float | None = None) -> "LizyWidget":
        """バックグラウンドスレッドで Tune を実行し完了まで待機する。
        失敗時は RuntimeError を raise する。"""

    # ── 結果取得 ───────────────────────────────────────────
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

メソッドチェーンをサポートする（`load()` / `set_target()` / `fit()` / `tune()` / `set_config()` は `self` を返す）。

`set_config()` / `load_config()` / `import_yaml` は UI の `patch_config` と同じ canonicalization 規約に従う。外部入力が partial dict / partial YAML でも、`config` traitlet に保持される値は Adapter が補完・正規化した canonical config とする。

```python
# 使用例
w = LizyWidget().load(df, target="price").load_config("lgbm.yaml")
w.fit()
summary = w.get_fit_summary()
```

---

## 5. UI 仕様

### 5.1 Widget 外観・レイアウト

セル出力にインライン表示される単一 Widget。横幅 100%（セル幅に追従）、高さ 620px 以上（最小高さ 620px。ウィジェット下端のドラッグハンドルでユーザーが任意に拡大可能）。

```
┌──────────────────────────────────────────────────┐
│ ⚡ LizyML Widget    lizyml v0.x.x    ● Idle      │  ← ヘッダー (40px)
├──────────────────────────────────────────────────┤
│ [▶ Data]  [ Model]  [ Results]                   │  ← タブバー (36px)
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
| Model | `status` が `data_loaded` 以降 |
| Results | `status` が `running` / `completed` / `failed` のいずれか、または `job_index > 0`（一度でも実行済み） |

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
│   Task    [binary][multiclass][regression] ⚡auto│
│                                                  │
│ ▸ Column Settings                                │
│   ┌─────────────┬──────┬──────┬────────────┐    │
│   │ Column      │ Uniq │ Excl │ Type       │    │
│   ├─────────────┼──────┼──────┼────────────┤    │
│   │ id          │ 1000 │ ☑    │ ── [ID]    │    │
│   │ const_col   │    1 │ ☑    │ ── [Const] │    │
│   │ age         │   50 │ ☐    │ [●Num][ Cat]│    │
│   │ city        │   15 │ ☐    │ [ Num][●Cat]│    │
│   └─────────────┴──────┴──────┴────────────┘    │
│                                                  │
│ ▸ Cross Validation                               │
│   Strategy [kfold][strat][group][time][purged]  │
│            [ - 5 + ] folds                        │
│                                                  │
│ ── Features: 15 cols (Numeric: 10, Categ.: 5) ── │
│ ── Excluded: 3 (ID: 1, Const: 1, Manual: 1)  ── │
└──────────────────────────────────────────────────┘
```

#### Target / Task

| 要素 | 動作 |
|------|------|
| Target | `df_info.columns` 全カラムのドロップダウン。選択時に Task 自動判定と Column 自動設定を実行 |
| Task | 自動判定結果を初期値とするセグメントボタン（`binary` / `multiclass` / `regression`）。1 クリックで切替可能 |

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

ユーザーは自動設定後にチェックボックスとセグメントボタンで各カラムを変更できる。
`excluded: true` の行は Type セグメントボタンを disabled 表示する。

#### Column Settings レイアウト（CSS Grid + minmax）

Column Settings は `<table>` ベースではなく CSS Grid ベースの行レイアウトへ置換する。

- 行コンテナ（例: `.lzw-columns-grid`）は `display: grid` を使い、`grid-template-columns: minmax(14rem, 2.4fr) minmax(4.5rem, 0.8fr) minmax(4.5rem, 0.8fr) minmax(12rem, 1.8fr)` で列幅を決定する
- ヘッダー行とデータ行で同一テンプレートを共有し、内容に応じて自動で伸縮する
- `Type` コントロールは列内で `width: 100%` とし、`minmax()` の範囲内でのみ拡縮する
- 親ラッパーは `overflow-x: auto` を維持し、Notebook の狭いセル幅では横スクロールに退避する
- 既存の列意味（Column/Uniq/Excl/Type）とデータフロー（`update_column`）は変更しない

#### Cross Validation

| 要素 | 説明 | 初期値 |
|------|------|------|
| Strategy | `kfold` / `stratified_kfold` / `group_kfold` / `time_series` / `purged_time_series` / `group_time_series` のセグメントボタン | Task 依存（`binary` / `multiclass` → `stratified_kfold`、`regression` → `kfold`） |
| Folds | 大型 `- / +` ステッパー付き数値入力（直接入力可） | `5` |
| Random state | `kfold` / `stratified_kfold` で表示 | `42` |
| Shuffle | `kfold` で表示 | `true` |
| Group column | `group_kfold` / `group_time_series` で表示。特徴量カラムのドロップダウン | `null` |
| Time column | `time_series` / `purged_time_series` / `group_time_series` で表示 | `null` |
| Gap | `time_series` / `group_time_series` で表示 | `0` |
| Purge gap | `purged_time_series` で表示 | `0` |
| Embargo | `purged_time_series` で表示 | `0` |
| Train/Test size max | 時系列系 split で表示 | `null` |

#### Data タブ要件（保存キーと初期値）

| Data タブ項目 | LizyML キー | 初期値 | 備考 |
|--------------|------------|--------|------|
| Target | `data.target` | `load(df, target=...)` 指定時はその値、未指定時は `null` | `null` のままでは実行不可 |
| Task | `task`（トップレベル） | Target 選択時に自動判定 | セグメントで手動変更可 |
| Excl（列除外） | `features.exclude` | 自動判定で ID/定数列を ON | 手動変更可 |
| Type=Categorical（非除外列） | `features.categorical` | 自動判定で設定 | `features.auto_categorical=true` を前提 |
| CV Strategy | `split.method` | Task 依存 | セグメントで手動変更。split alias は loader で正規化される |
| CV Folds | `split.n_splits` | `5` | 全 split 共通 |
| CV Random state | `split.random_state` | `42` | `kfold` / `stratified_kfold` のみ |
| CV Shuffle | `split.shuffle` | `true` | `kfold` のみ |
| Group column | `data.group_col` | `null` | Group 系 split で必須 |
| Time column | `data.time_col` | `null` | 時系列 split で必須 |
| Time gap | `split.gap` | `0` | `time_series` / `group_time_series` |
| Purge gap | `split.purge_gap` | `0` | `purged_time_series` |
| Embargo | `split.embargo` | `0` | `purged_time_series` |
| Train size max | `split.train_size_max` | `null` | 時系列 split で任意 |
| Test size max | `split.test_size_max` | `null` | 時系列 split で任意 |

#### Data パネル → Model 設定 自動反映

| Data タブの設定 | 反映先 |
|----------------|--------|
| Target | `data.target` |
| Task | `task` |
| Type = Categorical かつ Excl OFF | `features.categorical` |
| Excl = ON | `features.exclude` |
| CV Strategy | `split.method` |
| CV Folds | `split.n_splits` |
| Group column | `data.group_col` |
| Time column | `data.time_col` |
| Gap / Purge gap / Embargo | `split.gap` / `split.purge_gap` / `split.embargo` |
| Train/Test size max | `split.train_size_max` / `split.test_size_max` |

---

### 5.3 Model タブ（旧 Config タブ）

Model タブは Fit サブタブと Tune サブタブで構成する（実装コンポーネント名は `ConfigTab` を維持）。

```
┌──────────────────────────────────────────────────┐
│ [▶ Fit]  [ Tune]                  [━━ Fit ━━]   │  ← sticky（スクロールで隠れない）
├──────────────────────────────────────────────────┤
│ ▸ Model ─────────────────────────────────────── │
│   Model Type     lgbm （読み取り専用）            │
│                                                  │
│   ── Smart Params ──                             │
│   Auto Num Leaves  [●──]  ← lzw-toggle           │
│   Num Leaves Ratio  [ - 1.00 + ]  ← auto=ON 時  │
│   (auto=OFF 時: Num Leaves [ - 256 + ] に切替)  │
│   Min Data In Leaf Ratio  [ - 0.01 + ]           │  ← 常に表示
│   Min Data In Bin Ratio   [ - 0.01 + ]           │  ← 常に表示
│   Feature Weights  [──●]  ← lzw-toggle           │
│   (ON 時:)                                       │
│     ┌────────────────┬──────────────┬───┐        │
│     │ [col_a      ▼] │ [ - 1.5 + ]  │ × │        │  ← 列名セレクト + stepper
│     │ [col_b      ▼] │ [ - 2.0 + ]  │ × │        │
│     └────────────────┴──────────────┴───┘        │
│     [+ Add]                                      │
│   Balanced  [──●]  ← lzw-toggle (OFF=null,ON=true)│
│                                                  │
│   ── Model Params ──                             │
│   Objective    [binary|multiclass] ← segment btn │
│   Metric       [auc][binary_logloss][...]        │  ← chip (multi)
│   N Estimators [ - 1500 + ]                      │
│   Learning Rate[ - 0.001 + ]                     │
│   Max Depth    [ - 5 + ]                         │
│   Max Bin      [ - 511 + ]                       │
│   Feature Frac [ - 0.7 + ]                       │
│   Bagging Frac [ - 0.7 + ]                       │
│   Bagging Freq [ - 10 + ]                        │
│   Lambda L1    [ - 0.0 + ]                       │
│   Lambda L2    [ - 0.0 + ]                       │
│   First Metric Only  [●──]  ← toggle             │
│   Log Output (verbose)  [-1       ]              │
│                                                  │
│   ── Additional Params ──                        │
│   ┌──────────────────────┬──────────────┬───┐    │
│   │ [min_child_weight ▼] │ [ - 0.001 +] │ × │    │  ← パラメーター名セレクト + stepper
│   │ [extra_trees      ▼] │ [ - 1 +    ] │ × │    │
│   └──────────────────────┴──────────────┴───┘    │
│   [+ Add]                                        │
│                                                  │
│ ▸ Training ──────────────────────────────────── │
│   seed               [ - 42 + ]                 │
│   Early Stopping     [●──]                       │
│   (ON 時:)                                       │
│     Rounds           [ - 150 + ]                 │
│     Validation Ratio [ - 0.1 + ]                 │
│     Inner Validation [holdout   ▼]               │
│                                                  │
│ ▸ Evaluation ────────────────────────────────── │
│   metrics  [auc][logloss][f1][accuracy][...]    │  ← chip (multi)
│                                                  │
│ ▸ Calibration [●──] ────────────────────────── │  ← binary のみ。トグル左寄せ
│   (ON 時:)                                       │
│     method   [platt             ▼]               │
│     n_splits [5               ]                  │
│     params   [+ Add]                             │  ← KV エディタ（Additional Params 同形式）
│                                                  │
│ [Import YAML]  [Export YAML]  [Raw Config]       │
└──────────────────────────────────────────────────┘
```

#### フォーム動的生成

フォームフィールドは `backend_contract.config_schema` と `backend_contract.ui_schema` から動的に生成する。`config_schema` は値の型・制約・default を、`ui_schema` はセクション構成・表示順・widget 種別・option set・Search Space catalog・step 値を表す。

| JSON Schema 型 | フォームコンポーネント |
|----------------|---------------------|
| `number` / `integer` | `lzw-stepper`（大型 `- / +` ボタン + 直接入力） |
| `boolean` | `lzw-toggle`（スライドトグル） |
| `string` + `enum` | Select ドロップダウン |
| `string` | TextInput |
| `array` | タグ形式のマルチ入力 |
| `anyOf: [$ref, null]` | 非 null バリアントを展開して再帰的に解決（Pydantic Optional 型） |

- `default` / `const` / backend 固有 default → Adapter `initialize_config()` が canonical config に反映
- `description` / `title` / option set / step 値 → `backend_contract.ui_schema` から描画
- フォーム変更時に `patch_config` action をデバウンスして Python に送信
- Python 側は Service → Adapter `apply_config_patch()` → `validate_config()` の順で canonical config を再計算し、結果を `config` traitlet に反映
- discriminated union のような複合フィールド（例: `training.early_stopping.inner_valid`）で UI が表示用の selector state を持つ場合でも、Python へ送る payload は backend schema が要求する canonical object/null に正規化する
- `number` / `integer` はブラウザ標準スピナーを使わず、`lzw-stepper`（大型 `- / +`）で増減する
- `lzw-stepper` の数値入力欄は `width: 75px`（固定幅）を標準とする

**初期 backend（LizyML / LightGBM）が返す `widget_hints.step_map` の例:**

| フィールド | キー | step |
|-----------|------|------|
| Num Leaves Ratio | `model.num_leaves_ratio` | `0.05` |
| N Estimators | `model.params.n_estimators` | `100` |
| Learning Rate | `model.params.learning_rate` | `0.001` |
| Max Depth | `model.params.max_depth` | `1` |
| Feature Fraction | `model.params.feature_fraction` | `0.05` |
| Bagging Fraction | `model.params.bagging_fraction` | `0.05` |
| Bagging Freq | `model.params.bagging_freq` | `1` |
| Lambda L1 | `model.params.lambda_l1` | `0.0001` |
| Lambda L2 | `model.params.lambda_l2` | `0.0001` |
| Early Stopping Rounds | `training.early_stopping.rounds` | `50` |
| Validation Ratio | `training.early_stopping.validation_ratio` | `0.05` |

#### Fit サブタブ要件（初期 backend contract 例: LightGBM）

Fit / Tune サブタブバー（`[Fit] [Tune]` ボタン + 実行ボタン）はコンテンツエリアの上端に `position: sticky; top: 0` で固定し、スクロールしても常に表示する。

| UI 項目 | キー | 初期値 | 備考 |
|--------|------|--------|------|
| Config Version | `config_version` | `1` | 固定値（読み取り専用） |
| Model Type | `model.name` | `"lgbm"` | 初期 backend contract では読み取り専用 |
| Params | `model.params` | LightGBM backend default 一式（下表参照） | Adapter `initialize_config()` が default を返し、UI は `backend_contract.ui_schema` の field 定義に従って描画する。task 変更時の objective・metric 切替も Adapter が canonical config と option set を再計算する |
| Auto Num Leaves | `model.auto_num_leaves` | `true` | `true` 時は `params.num_leaves` と排他 |
| Num Leaves Ratio | `model.num_leaves_ratio` | `1.0` | `auto_num_leaves=true` 時のみ表示。`0 < value <= 1` |
| Num Leaves | `model.params.num_leaves` | `256` | `auto_num_leaves=false` 時のみ表示。表示条件は contract の visibility rule に従う |
| Min Data In Leaf Ratio | `model.min_data_in_leaf_ratio` | `0.01` | `lzw-stepper` を常に表示。`params.min_data_in_leaf` と排他 |
| Min Data In Bin Ratio | `model.min_data_in_bin_ratio` | `0.01` | `lzw-stepper` を常に表示。`params.min_data_in_bin` と排他 |
| Feature Weights | `model.feature_weights` | `null` | `lzw-toggle` ON/OFF。OFF=`null`（無効）。ON 時は列名セレクト（`df_info.columns`）+ 重み `lzw-stepper`（初期値 `1.0`）のペアを複数行追加可能。指定時は全値 `> 0` |
| Balanced | `model.balanced` | `null` | `lzw-toggle` ON/OFF。OFF=`null`（task から自動判定）、ON=`true`（強制バランシング） |
| Additional Params | `model.params.*` | — | `parameter_hints` 定義済みキー以外の任意 LightGBM パラメータ。パラメーター名 `<select>`（`ui_schema.additional_params` から候補供給）+ `lzw-stepper` で追加。各行に `×` 削除ボタン。[+ Add] で行追加 |
| Seed | `training.seed` | `42` | モデル random_state の既定元 |
| Early Stopping Enabled | `training.early_stopping.enabled` | `true` | `false` で inner valid 無効 |
| Early Stopping Rounds | `training.early_stopping.rounds` | `150` | 大型 `- / +` ステッパー付き数値入力 |
| Validation Ratio | `training.early_stopping.validation_ratio` | `0.1` | `inner_valid` 非明示時の shorthand |
| Inner Validation | `training.early_stopping.inner_valid` | `null`（UI 初期表示は `holdout`） | `<select>` で `holdout` / `group_holdout` / `time_holdout` を表示。`null` 送信時は backend が holdout を使用するため、UI の初期選択は `holdout` とする。canonical value は `{method: ...}` を含む object または `null` とし、表示専用ラベル（例: `fold_0`）を保存してはいけない |
| Metrics | `evaluation.metrics` | `[]` | 初期 backend contract の task 別 option set をチップボタン表示。空配列 = task 別 runtime default |
| Calibration Toggle | `calibration` | `null`（OFF） | binary のみ有効。Accordion ヘッダ右端の `lzw-toggle` で ON/OFF。トグルクリックは Accordion 開閉と独立（`e.stopPropagation()`）。ON 時は `{ method, n_splits, params }` を設定 |
| Calibration Method | `calibration.method` | `"platt"` | ON 時のみ表示 |
| Calibration Folds | `calibration.n_splits` | `5` | ON 時のみ表示 |
| Calibration Params | `calibration.params` | `{}` | ON 時のみ表示。Additional Params と同じ KV エディタ形式（パラメーター名 `<select>` + `lzw-stepper`）。`ui_schema.additional_params` から候補供給 |
| Log Output | `model.params.verbose` | `-1` | Model セクション下部に独立した大型 `- / +` ステッパー付き数値入力として表示 |
| Output Directory | `output_dir` | `null` | 任意 |

**初期 backend（LizyML / LightGBM）が `initialize_config()` で返す `model.params` 例:**

task 非依存項目も task 依存項目も backend default として Adapter が所有する。Service は task / target / split など Data タブ由来 state を保持し、Adapter hook を呼び出して canonical config を再計算する。

| キー | 初期値 | task 依存 |
|------|--------|-----------|
| `objective` | regression=`huber`, binary=`binary`, multiclass=`multiclass` | ✓ | セグメントボタン。候補は `ui_schema.option_sets.objective[task]` |
| `metric` | regression=`["huber","mae","mape"]`, binary=`["auc","binary_logloss"]`, multiclass=`["auc_mu","multi_logloss"]` | ✓ |
| `n_estimators` | `1500` | |
| `learning_rate` | `0.001` | |
| `max_depth` | `5` | |
| `max_bin` | `511` | |
| `feature_fraction` | `0.7` | |
| `bagging_fraction` | `0.7` | |
| `bagging_freq` | `10` | |
| `lambda_l1` | `0.0` | |
| `lambda_l2` | `0.000001` | |
| `first_metric_only` | `false` | |
| `verbose` | `-1` | UI では params KVEditor から独立した「Log Output」項目として表示 |
| `num_leaves` | `256` | `auto_num_leaves=false` 時のみ補完 |

`auto_num_leaves=false` 時は `params.num_leaves` を補完する（デフォルト `256`）。`auto_num_leaves=true` 時は `params.num_leaves` を設定すると LizyML がバリデーションエラーを返すため、Adapter が canonicalization の中で params から除去する。

ユーザーが params の値を変更・追加・削除した場合はその値が優先される。`model.params` を完全に空にした場合は LizyML が runtime default を補完する（後退互換）。

#### Tune サブタブ

Tune タブは 3 セクション構成: Tuning Settings → Search Space（Model Params + Training 統合）→ Evaluation。Fit タブの設定には依存せず、Search Space の Fixed 値と Evaluation が Tune 実行時の全設定を決定する。

```
┌──────────────────────────────────────────────────┐
│ [ Fit]  [▶ Tune]                  [━━ Tune ━━]  │  ← sticky
├──────────────────────────────────────────────────┤
│ ▸ Tuning Settings ──────────────────────────── │
│   n_trials      [ - 50 + ]                       │
│                                                  │
│ ▸ Search Space ──────────────────────────────── │
│   ┌──────────────────┬────────────────┬──────────────────┐
│   │ Param            │ Mode           │ Config           │
│   ├──────────────────┼────────────────┼──────────────────┤
│   │ ── Model Params ─┼────────────────┼──────────────────┤
│   │ objective        │ [Fixed|Choice] │ [binary       ▼] │
│   │ metric           │ [Fixed|Choice] │ [auc][bin_l][..] │
│   │ n_estimators     │ [Fixed|Range ] │ [ - 1500 +     ] │
│   │ learning_rate    │ [Fixed|Range ] │ [ - 0.001 +    ] │
│   │ max_depth        │ [Fixed|Range ] │ [ - 5 +        ] │
│   │ max_bin          │ [Fixed|Range ] │ [ - 511 +      ] │
│   │ feature_fraction │ [Fixed|Range ] │ [ - 0.7 +      ] │
│   │ bagging_fraction │ [Fixed|Range ] │ [ - 0.7 +      ] │
│   │ bagging_freq     │ [Fixed|Range ] │ [ - 10 +       ] │
│   │ lambda_l1        │ [Fixed|Range ] │ [ - 0.0 +      ] │
│   │ lambda_l2        │ [Fixed|Range ] │ [ - 0.000001 + ] │
│   │ first_metric_only│ [Fixed|Choice] │ [●──]            │
│   │ verbose          │ [Fixed|Range ] │ [ - -1 +       ] │
│   │ ── Smart Params ─┼────────────────┼──────────────────┤
│   │ auto_num_leaves  │ [Fixed|Choice] │ [●──]            │
│   │ num_leaves_ratio │ [Fixed|Range ] │ [ - 1.0 +      ] │
│   │ num_leaves       │ [Fixed|Range ] │ [ - 256 +      ] │
│   │ min_data_in_l... │ [Fixed|Range ] │ [ - 0.01 +     ] │
│   │ min_data_in_b... │ [Fixed|Range ] │ [ - 0.01 +     ] │
│   │ feature_weights  │ Fixed          │ [──●]            │
│   │ balanced         │ [Fixed|Choice] │ [──●]            │
│   │ ── Training ─────┼────────────────┼──────────────────┤
│   │ seed             │ Fixed          │ [ - 42 +       ] │
│   │ early_stop.enable│ Fixed          │ [●──]            │
│   │ early_stop.rounds│ [Fixed|Range ] │ [ - 150 +      ] │
│   │ validation_ratio │ [Fixed|Range ] │ [ - 0.1 +      ] │
│   │ inner_valid      │ Fixed          │ [holdout     ▼]  │
│   ├──────────────────┼────────────────┼──────────────────┤
│   │ [+ Add ▼]        │                │                  │
│   └──────────────────┴────────────────┴──────────────────┘
│                                                  │
│ ▸ Evaluation ────────────────────────────────── │
│   Optimization Metric                            │
│     [ auc | logloss | f1 | accuracy | ...]      │  ← segment (single)
│   Additional Metrics                             │
│     [logloss][f1][accuracy][...]                 │  ← chip (multi, 任意)
│                                                  │
│ [Import YAML]  [Export YAML]  [Raw Config]       │
└──────────────────────────────────────────────────┘
```

#### Tune サブタブ要件（初期 backend contract 例: Optuna）

| UI 項目 | キー | 初期値 | 備考 |
|--------|------|--------|------|
| Tune Toggle | `tuning` | `null` | Tune 設定未使用時は `null` |
| Number of Trials | `tuning.optuna.params.n_trials` | `50` | 大型 `- / +` ステッパー付き数値入力 |
| Search Space (Model Params) | `tuning.model_params` + `tuning.optuna.space` | Fit の `model.params` からコピー | Fixed 値は `tuning.model_params` に、Range/Choice は `tuning.optuna.space` に格納。Widget-only フィールド（`strip_for_backend` で除去） |
| Search Space (Smart Params) | `tuning.model_params` + `tuning.optuna.space` | Fit の Smart Params からコピー | `feature_weights` は Fixed 限定。他は Fixed / Range / Choice 選択可能 |
| Search Space (Training) | `tuning.training` + `tuning.optuna.space` | Fit の `training` からコピー | `seed` / `enabled` / `inner_valid` は Fixed 限定。`rounds` / `validation_ratio` は Fixed / Range 選択可能。Widget-only フィールド |
| Optimization Metric | `tuning.evaluation.metrics[0]` | task 別の先頭 metric | セグメントボタン（単一選択）。候補は `ui_schema.option_sets.metric[task]`。`direction` は Adapter が自動決定 |
| Additional Metrics | `tuning.evaluation.metrics[1..]` | `[]` | チップボタン（複数選択、任意）。Optimization Metric を除いた候補。LizyML は全メトリックを計算するが Optuna objective には使用しない |

Search Space は Tune における **model params と training params のベースライン値 + 探索空間の統合管理場所**となる。`backend_contract.ui_schema.search_space_catalog` から pre-populate された行を **Model Params**・**Smart Params**・**Training** のサブグループ見出しで視覚的に区切る。[+ Add] で任意パラメータ（`ui_schema.additional_params` から `<select>` で選択）を行追加できる。

Fixed 値の初期値は `initialize_config` 時に Fit の現在値（`model.params` / Smart Params / `training`）からコピーする。以降は Fit と独立して編集可能。Search Space に含まれないパラメータは Tune 実行時に backend default を使用する（Fit の値にはフォールバックしない）。

Range/Choice が 0 件のときに empty `space={}` を許容するかどうかは `backend_contract.capabilities` で決める。初期 backend では empty `space={}` を UI / Python API の両方で許容する。

Search Space の `metric`（`tuning.optuna.space.metric`）は、Optuna 最適化指標ではなく **LightGBM パラメータ `model.params.metric` の探索軸**として扱う。Choice で設定した候補は trial ごとに 1 つ選択され、他の trial params と同様に estimator へ渡される。

Mode 列は **ドロップダウンではなくセグメントボタン**（Fixed / Range / Choice）で表示する。行ごとに利用可能な候補のみ表示し、1 クリックでモード切替できるようにする。

#### Search Space レイアウト（CSS Grid + minmax）

Search Space も `<table>` ベースではなく CSS Grid に置換し、列幅は `minmax()` で自動調整する。

- 行コンテナ（例: `.lzw-search-space-grid`）は `display: grid` + `grid-template-columns: minmax(14rem, 2.6fr) minmax(9rem, 1.2fr) minmax(16rem, 2.2fr)` を使用する
- `Config` 列の内部コントロール（select / stepper / chip group）は `width: 100%` を基本とし、列幅制約は `minmax()` 側で管理する
- `Range` モードの `low` / `high` はどちらも `lzw-stepper`（大型 `- / +`）を使用し、直接入力欄は `width: 75px` を維持する
- 親の `.lzw-search-space` は `overflow-x: auto` を維持し、Notebook の狭い幅では横スクロールへ退避する
- 既存の `tuning.optuna.space` 契約（`type` ベース）と Mode 切替ロジックは変更しない

Search Space の全パラメータ一覧（初期 backend contract 例）:

| Param | 種別 | 初期モード | 初期値 / Fixed 値 | 選択可能 Mode |
|-------|------|-----------|-------------------|--------------|
| `objective` | LGBMParam | Fixed | task 依存 | Fixed / Choice | Fixed 時はセグメントボタン（`option_sets.objective[task]`）。Choice 時はチップボタン（複数選択可）。`config.choices` に選択した `string[]` を格納 |
| `metric` | LGBMParam | Fixed | task 依存 | Fixed / Choice | Choice 時は `backend_contract.ui_schema.option_sets.metric[task]` のチップボタン（複数選択可）。`config.choices` に選択した `string[]` を格納。`tuning.optuna.space.metric` として保存され、trial ごとに `model.params.metric` として適用される |
| `n_estimators` | LGBMParam | Fixed | `1500` | Fixed / Range |
| `learning_rate` | LGBMParam | Fixed | `0.001` | Fixed / Range |
| `max_depth` | LGBMParam | Fixed | `5` | Fixed / Range |
| `max_bin` | LGBMParam | Fixed | `511` | Fixed / Range |
| `feature_fraction` | LGBMParam | Fixed | `0.7` | Fixed / Range |
| `bagging_fraction` | LGBMParam | Fixed | `0.7` | Fixed / Range |
| `bagging_freq` | LGBMParam | Fixed | `10` | Fixed / Range |
| `lambda_l1` | LGBMParam | Fixed | `0.0` | Fixed / Range |
| `lambda_l2` | LGBMParam | Fixed | `0.000001` | Fixed / Range |
| `first_metric_only` | LGBMParam | Fixed | `false` | Fixed / Choice |
| `auto_num_leaves` | LGBMConfig | Fixed | `true` | Fixed / Choice |
| `num_leaves_ratio` | LGBMConfig | Fixed | `1.0` | Fixed / Range | `auto_num_leaves=true` 時のみ表示 |
| `num_leaves` | LGBMParam | Fixed | `256` | Fixed / Range / Choice | `auto_num_leaves=false` 時、または `auto_num_leaves` が Choice モードの時に表示 |
| `min_data_in_leaf_ratio` | LGBMConfig | Fixed | `0.01` | Fixed / Range |
| `min_data_in_bin_ratio` | LGBMConfig | Fixed | `0.01` | Fixed / Range |
| `feature_weights` | LGBMConfig | Fixed | `null` | Fixed | Fixed 限定。`lzw-toggle` で ON/OFF。Tune の `resolve_smart_params_from_dict` には feature_weights ロジックがないため探索不可 |
| `balanced` | LGBMConfig | Fixed | `null` | Fixed / Choice |
| `seed` | Training | Fixed | `42` | Fixed | Fixed 限定。`lzw-stepper` |
| `early_stopping.enabled` | Training | Fixed | `true` | Fixed | Fixed 限定。`lzw-toggle` |
| `early_stopping.rounds` | Training | Fixed | `150` | Fixed / Range | Tune 探索空間の `early_stopping_rounds` に対応 |
| `validation_ratio` | Training | Fixed | `0.1` | Fixed / Range | Tune 探索空間の `validation_ratio` に対応 |
| `inner_valid` | Training | Fixed | `null`（UI 初期表示 `holdout`） | Fixed | Fixed 限定。`<select>` で holdout / group_holdout / time_holdout。`null` 送信時は backend が holdout を使用 |

`tuning.optuna.space = {}` のときの LizyML backend default Search Space（参考）:

| Param | Type | Range / Choices | Log |
|-------|------|------------------|-----|
| `objective` | `categorical` | task 依存（regression=`huber,fair` / binary=`binary` / multiclass=`multiclass,multiclassova`） | - |
| `n_estimators` | `int` | `600 .. 2500` | `false` |
| `early_stopping_rounds` | `int` | `40 .. 240` | `false` |
| `validation_ratio` | `float` | `0.1 .. 0.3` | `false` |
| `learning_rate` | `float` | `0.0001 .. 0.1` | `true` |
| `max_depth` | `int` | `3 .. 12` | `false` |
| `feature_fraction` | `float` | `0.5 .. 1.0` | `false` |
| `bagging_fraction` | `float` | `0.5 .. 1.0` | `false` |
| `num_leaves_ratio` | `float` | `0.5 .. 1.0` | `false` |
| `min_data_in_leaf_ratio` | `float` | `0.01 .. 0.2` | `false` |

Search Space の Mode 選択肢は `backend_contract.ui_schema.search_space_catalog` の field type / capability で決まる:

| パラメータ型 | 選択可能な Mode | Mode UI | Fixed 時の Config UI | Choice 時の Config UI |
|------------|---------------|---------|----------------------|------------------------|
| `float` / `integer` | Fixed / Range | セグメントボタン（2分割） | `lzw-stepper`（大型 `- / +`） | - |
| `enum` / `string` （objective 等） | Fixed / Choice | セグメントボタン（2分割） | `<select>` | contract の option set を使うチップボタン（複数選択可） |
| `array` + items.enum （metric 等） | Fixed / Choice | セグメントボタン（2分割） | チップボタン（複数選択可） | contract の option set を使うチップボタン（複数選択可） |
| `boolean` | Fixed / Choice | セグメントボタン（2分割） | `lzw-toggle` | - |

Range モードの `low` / `high` 入力も同じ `lzw-stepper` を使い、`step` 値に従って増減する。入力欄幅は `75px` を使用する。

**Fit / Tune ボタン有効条件:**

| 条件 | Fit | Tune |
|------|-----|------|
| `status` が `data_loaded` 以降 | 必須 | 必須 |
| `df_info.target` が設定済み | 必須 | 必須 |
| `config` が `backend_contract.config_schema` を満たす | 必須 | 必須 |
| `tuning.optuna.space` 空許可 | 不要 | `backend_contract.capabilities.tune.allow_empty_space` に従う |

> **Note:** 初期 backend では `tuning.optuna.space = {}` を UI / Python API ともに許容し、`prepare_run_config(job_type="tune")` が backend default search space を補完する。

#### Model Config Import / Export

| 操作 | 説明 |
|------|------|
| Import YAML | YAML / JSON ファイルを Widget 内のファイルピッカーで選択して読み込む。`data` / `features` / `split` / `task` は Data タブに反映、その他は Model フォームに反映 |
| Export YAML | 現在の全設定（Data + Model 両タブ）を YAML でダウンロード |
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
| `roc-curve` | binary / multiclass |
| `calibration` | binary + calibration 有効 |
| `probability-histogram` | binary + calibration 有効 |

セレクタ変更時に `request_plot` action を送信。Python がプロット生成し `msg:custom` で返す。

**Accordion セクション:**

| セクション | 内容 | 表示条件 |
|-----------|------|---------|
| Feature Importance | 特徴量重要度の棒グラフ（Plotly） | 常時 |
| Fold Details | Fold 別メトリクスとデータサイズのテーブル | CV 時のみ。Fold 別スコアのリアルタイム表示はバックエンドの Fold コールバック API 対応に依存する。LizyML 初期版では `fold_results` は空のため、完了後の集計値のみ表示する |
| Parameters | 使用ハイパーパラメータ一覧 | 常時 |

#### Tune 完了

Fit 完了と同じ構成に加え、先頭に探索結果を追加する。

```
Optimization History → Best Params → [Apply to Fit ▸]
→ Score → Learning Curve → Plots → Accordion(Trial Results / Feature Importance / Fold Details / Parameters)
```

**Optimization History:** Trial 番号 vs スコアの収束プロット（Plotly）。

**Best Params + Apply to Fit:** Best Params テーブルと、Fit サブタブへ設定を同期して切り替えるボタン。

Search Space で `metric` を探索対象に含めた場合、`best_params.metric` は LightGBM `model.params.metric` として解釈する。

Apply to Fit の同期ルール:

1. Tune 実行時に使用した config スナップショットを復元する
2. `best_params` を `model.params` に上書き適用する
3. Model タブ Fit サブタブへ切り替える

このとき Fit 画面の全パラメータ（`model` / `training` / `evaluation` / `calibration` / `output_dir` 等）は Tune 実行時設定と一致する。

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
| `TARGET_ERROR` | Target 設定時のエラー |
| `TASK_ERROR` | Task 設定時のエラー |
| `COLUMN_ERROR` | Column 更新時のエラー |
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

### 6.3 Config 契約

完全疎結合化後の canonical config は **Python 側のみを正**とする。UI は `config` traitlet のスナップショットを表示し、編集要求は `patch_config` action で送る。backend 固有の default / option set / Search Space catalog / widget hint はすべて `backend_contract` から供給される。

#### 契約ルール

1. UI は full config dict を直接生成・送信しない。backend config の変更は `patch_config` action のみを使う
2. `backend_contract.config_schema` が型と値制約の正、`backend_contract.ui_schema` が表示・入力 control・option set・step 値の正とする
3. `WidgetService` は `task` / `data.*` / `features.*` / `split.*` を保持し、Adapter が返す canonical config と merge して最終実行 config を構築する
4. Adapter は `initialize_config()` / `apply_config_patch()` / `prepare_run_config()` により backend 固有 default・visibility・Tune 補完・legacy 正規化を一元管理する
5. Legacy raw config import を含む最終受理判定は Adapter `validate_config()` が行う

#### Canonical config 生成フロー

1. `load()` または backend 切替時に WidgetService が Adapter `get_backend_contract()` と `initialize_config(task=...)` を取得し、`backend_contract` / `config` traitlet を更新する
2. UI 編集時は `patch_config` を送信し、WidgetService が Adapter `apply_config_patch()` を呼んで canonical config を再計算する
3. Fit / Tune 実行時は WidgetService が Data タブ state を merge し、Adapter `prepare_run_config(..., job_type=...)` を通して最終 config を作る
4. 最終 config は Adapter `validate_config()` を通して runtime へ渡す

#### Patch Operation 契約

| `op` | `path` | `value` | 説明 |
|---|---|---|---|
| `set` | dot path | 必須 | 値を上書きまたは作成する |
| `unset` | dot path | 省略可 | key を削除し、backend default または未設定状態へ戻す |
| `merge` | dot path | `dict` | object を shallow merge する |

UI は Search Space の `mode=Fixed/Range/Choice` のような表示用 state をローカルに保持できるが、Python へ送る payload は `ConfigPatchOp` に正規化されたものだけとする。

#### Table A: Data / Features / Split（Service 主責務）

| Config Path | 型 | 必須 | 生成元 | 変換レイヤー | エラーコード |
|---|---|---|---|---|---|
| `task` | `"binary" \| "multiclass" \| "regression"` | Fit/Tune 時必須 | Service 自動判定 or UI `set_task` | WidgetService → `prepare_run_config()` | `VALIDATION_ERROR` |
| `data.target` | `string` | Yes | UI `set_target` | WidgetService → `prepare_run_config()` | `NO_TARGET` |
| `data.group_col` | `string \| null` | group 系 CV 時 | UI `update_cv` | WidgetService → `prepare_run_config()` | `VALIDATION_ERROR` |
| `data.time_col` | `string \| null` | time 系 CV 時 | UI `update_cv` | WidgetService → `prepare_run_config()` | `VALIDATION_ERROR` |
| `features.categorical` | `string[]` | No | Service 自動判定 + UI | WidgetService → `prepare_run_config()` | — |
| `features.exclude` | `string[]` | No | Service 自動判定 + UI | WidgetService → `prepare_run_config()` | — |
| `split.method` | `string` | Yes | UI `update_cv` | WidgetService → `prepare_run_config()` | `VALIDATION_ERROR` |
| `split.n_splits` | `int >= 2` | Yes | UI `update_cv` | WidgetService → `prepare_run_config()` | `VALIDATION_ERROR` |

#### Table B: Backend 固有 Config（Adapter 主責務）

| Config Path | 型 | 必須 | 生成元 | 変換レイヤー | エラーコード |
|---|---|---|---|---|---|
| `config_version` | `int` | Yes | Adapter `initialize_config()` | Adapter `initialize_config()` / `prepare_run_config()` | `VALIDATION_ERROR` |
| `model.name` | backend 定義の const | Yes | Adapter `initialize_config()` | Adapter `initialize_config()` / `apply_config_patch()` | `VALIDATION_ERROR` |
| `model.params` | `dict` | Yes | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | `VALIDATION_ERROR` |
| `model.auto_num_leaves` | `bool` | backend 依存 | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | — |
| `training.seed` | `int` | backend 依存 | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | — |
| `training.early_stopping.enabled` | `bool` | backend 依存 | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | — |
| `training.early_stopping.inner_valid` | `HoldoutInnerValidConfig \| GroupHoldoutInnerValidConfig \| TimeHoldoutInnerValidConfig \| null` | backend 依存 | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` / `prepare_run_config()` | `VALIDATION_ERROR` |
| `evaluation.metrics` | `string[]` | No | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | — |
| `calibration` | `dict \| null` | No | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` | — |

#### Table C: Tuning（Adapter 主責務 + Service 実行準備）

| Config Path | 型 | 必須 | 生成元 | 変換レイヤー | エラーコード |
|---|---|---|---|---|---|
| `tuning.optuna.params.n_trials` | `int >= 1` | Tune 時 backend 依存 | Adapter default + UI `patch_config` | Adapter `apply_config_patch()` / `prepare_run_config()` | `VALIDATION_ERROR` |
| `tuning.optuna.space.<param>` (Range) | `{type: "float"\|"int", low: number, high: number, log?: bool}` | capability に依存 | UI SearchSpace local mode → `patch_config` | UI が `mode→type` へ正規化し、Adapter `apply_config_patch()` が最終 canonicalize | `VALIDATION_ERROR` |
| `tuning.optuna.space.<param>` (Choice) | `{type: "categorical", choices: any[]}` | capability に依存 | UI SearchSpace local mode → `patch_config` | UI が `mode→type` へ正規化し、Adapter `apply_config_patch()` が最終 canonicalize | `VALIDATION_ERROR` |
| `tuning.optuna.space.<param>` (Fixed) | key 削除 | — | UI SearchSpace local mode → `patch_config` | UI が `unset` を送り、Adapter `apply_config_patch()` が default へ戻す | — |
| `tuning.model_params` | `dict` | No | Search Space Fixed model param 値。初期値は Fit の `model.params` からコピー | Widget-only（`strip_for_backend` で除去）。`prepare_run_config(tune)` で `model.params` に置換 | — |
| `tuning.training` | `dict` | No | Search Space Fixed training 値。初期値は Fit の `training` からコピー | Widget-only（`strip_for_backend` で除去）。`prepare_run_config(tune)` で `training` に置換 | — |
| `tuning.evaluation` | `dict` | No | Tune Evaluation セクション。`metrics[0]` = Optimization Metric | Widget-only（`strip_for_backend` で除去）。`prepare_run_config(tune)` で `evaluation` に置換。`direction` は Adapter が自動決定 | — |

#### 禁止フォーマット（Adapter `apply_config_patch()` / `validate_config()` が拒否）

| パターン | エラー種別 | 例 |
|---|---|---|
| JS から full config dict を `update_config` 相当で送信 | `config_transport` | `{"config": {...}}` |
| `mode` キーあり・`type` キーなし | `search_space_format` | `{"mode": "range", "low": 0.01, "high": 0.1}` |
| `type` が有効値以外 | `invalid_space_type` | `{"type": "range", "low": 0.01, "high": 0.1}` |

有効な `type` 値: `"float"`, `"int"`, `"categorical"`

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
│   ├── ColumnTable.tsx    # Column Settings グリッド
│   ├── DynForm.tsx        # backend_contract.config_schema/ui_schema → フォーム動的生成
│   ├── SearchSpace.tsx    # backend_contract.ui_schema.search_space_catalog → Tune の Search Space 描画
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
  --bundle --format=esm --jsx=automatic --jsx-import-source=preact \
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
  --jsx=automatic --jsx-import-source=preact \
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

### 8.5 バージョニング

[Semantic Versioning 2.0.0](https://semver.org/) に準拠する。

```
MAJOR.MINOR.PATCH   例: 0.1.0, 0.2.0, 1.0.0
```

#### Alpha 期間 (`0.x.y`)

`1.0.0` に達するまでは public API は安定保証しない。

| バージョン変更 | 条件 | 例 |
|---------------|------|-----|
| `0.MINOR+1.0` | 機能追加・破壊的変更 | 新タブ追加、traitlets 変更、Adapter Protocol 変更 |
| `0.MINOR.PATCH+1` | バグ修正・ドキュメント修正 | メトリクス修正、UI 調整 |

#### Stable 期間 (`>=1.0.0`)

| バージョン変更 | 条件 |
|---------------|------|
| `MAJOR+1.0.0` | 後方互換性を破る変更（破壊的変更） |
| `MINOR+1.0` | 後方互換で機能追加 |
| `PATCH+1` | 後方互換でバグ修正 |

#### バージョン取得

`hatch-vcs` が git タグから自動取得する。手動でのバージョン指定は不要。

```python
import lizyml_widget
print(lizyml_widget.__version__)  # "0.1.0"
```

| 状態 | バージョン例 |
|------|-------------|
| タグ `v0.1.0` が HEAD | `0.1.0` |
| タグから 3 コミット先 | `0.1.1.dev3+gabc1234` |

#### リリースフロー

1. develop → main にマージ
2. GitHub Release を作成（タグ `vX.Y.Z`）
3. `publish.yml` が自動起動 → TestPyPI → PyPI

#### 破壊的変更ポリシー

- Alpha 期間中でも HISTORY.md に Proposal を記録してから実施
- Stable 期間では deprecation warning を 1 MINOR バージョン以上出してから削除

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
