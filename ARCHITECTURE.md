# ARCHITECTURE.md

> 本ドキュメントは実装コードから導出したアーキテクチャの実態記録である。
> 仕様の正は BLUEPRINT.md（§1 優先順位を参照）。

## 1. システム概要

LizyML-Widget は Jupyter Notebook / Google Colab / VS Code Notebooks 向けの ML 補助 UI Widget。
DataFrame を渡してコードなしで LizyML の Fit・Tune・Inference をセル内で操作する。

**テックスタック:**
- Python: anywidget + traitlets（状態同期）、threading.Thread（バックグラウンド実行）
- TypeScript: Preact + esbuild（ESM バンドル）、Plotly.js（CDN 動的ロード）

## 2. レイヤーアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  UI Layer (Preact)                                      │
│  js/src/  23 files, ~4,100 LOC                          │
│  tabs/ + components/ + hooks/                           │
├────────────── anywidget traitlets + msg:custom ──────────┤
│  Widget Layer                                           │
│  widget.py  ~680 LOC                                    │
│  LizyWidget (anywidget.AnyWidget)                       │
├─────────────────────────────────────────────────────────┤
│  Service Layer                                          │
│  service.py  ~640 LOC                                   │
│  WidgetService                                          │
├─────────────────────────────────────────────────────────┤
│  Adapter Layer                                          │
│  adapter.py + adapter_params.py + adapter_schema.py     │
│  + adapter_contract.py  ~1,700 LOC                      │
│  BackendAdapter (Protocol) + LizyMLAdapter              │
├─────────────────────────────────────────────────────────┤
│  ML Backend                                             │
│  lizyml[plots,tuning,calibration,explain]               │
└─────────────────────────────────────────────────────────┘
```

### 2.1 各レイヤーの責務

| レイヤー | 責務 | 禁止事項 |
|---------|------|---------|
| **UI** | traitlets 読み書き、`backend_contract` の generic 描画 | ML ライブラリ直接呼出、backend 固有カタログのハードコード |
| **Widget** | traitlets 定義、Action ディスパッチ、スレッド管理 | ML ロジック記述、Service private 属性アクセス |
| **Service** | Data 状態管理、実行前提判定、config 構築 | backend 固有 default / option set / search space catalog |
| **Adapter** | ML ライブラリ呼出、共通型変換、config lifecycle | Widget / traitlets の知識 |

### 2.2 Adapter サブモジュール構成

実装では Adapter 層は4ファイルに分割されている：

| ファイル | LOC | 責務 |
|---------|-----|------|
| `adapter.py` | ~770 | BackendAdapter Protocol 定義 + LizyMLAdapter 本体 |
| `adapter_params.py` | ~190 | タスク別パラメータセット、メトリクスマッピング、ハイパーパラメータ分類 |
| `adapter_schema.py` | ~360 | JSON Schema バリデーション、config 準備、search space デフォルト |
| `adapter_contract.py` | ~370 | UI Schema・capabilities ビルダー、BackendContract 仕様 |

## 3. 状態同期（traitlets）

### 3.1 traitlets 一覧

| traitlet | 型 | 方向 | 用途 |
|----------|---|------|------|
| `backend_info` | Dict | P→JS | バックエンド名・バージョン |
| `df_info` | Dict | P→JS | DataFrame メタデータ（shape, target, task, columns, cv, feature_summary） |
| `backend_contract` | Dict | P→JS | config_schema, ui_schema, capabilities |
| `config` | Dict | P→JS | Widget 所有の正規化 config |
| `status` | Unicode | P→JS | `idle` / `data_loaded` / `running` / `completed` / `failed` |
| `job_type` | Unicode | P→JS | `fit` / `tune` / `""` |
| `job_index` | Int | P→JS | ジョブカウンタ（キャンセル追跡用） |
| `progress` | Dict | P→JS | `{current, total, message, fold_results?}` |
| `elapsed_sec` | Float | P→JS | 経過秒数 |
| `fit_summary` | Dict | P→JS | Fit 結果（metrics, fold_count, fold_details, params） |
| `tune_summary` | Dict | P→JS | Tune 結果（best_params, best_score, trials, metric_name, direction） |
| `available_plots` | List | P→JS | 利用可能プロット種別 |
| `inference_result` | Dict | P→JS | 推論結果 |
| `error` | Dict | P→JS | `{code, message, details?}` |
| `action` | Dict | JS→P | `{type, payload, _ts}` — UI コマンド |

### 3.2 Action 一覧

| type | payload | 説明 |
|------|---------|------|
| `set_target` | `{target}` | ターゲット列設定 |
| `set_task` | `{task}` | タスク手動変更 |
| `update_column` | `{name, excluded, col_type}` | 列設定変更 |
| `update_cv` | `{strategy, n_splits, ...}` | CV 設定変更 |
| `patch_config` | `{ops: [{op, path, value}]}` | Config パッチ（set/unset/merge） |
| `fit` | `{}` | Fit 実行 |
| `tune` | `{}` | Tune 実行 |
| `cancel` | `{}` | ジョブキャンセル |
| `request_plot` | `{plot_type}` | プロット要求 |
| `request_inference_plot` | `{plot_type}` | 推論プロット要求 |
| `run_inference` | `{return_shap}` | 推論実行 |
| `apply_best_params` | `{params}` | Tune ベストパラメータ適用 |
| `import_yaml` | `{content}` | YAML インポート |
| `export_yaml` | `{}` | YAML エクスポート |
| `raw_config` | `{}` | デバッグ用 config 取得 |

### 3.3 msg:custom メッセージ

| type | 方向 | 用途 |
|------|------|------|
| `plot_data` | P→JS | Plotly JSON 配信 |
| `plot_error` | P→JS | プロットエラー |
| `yaml_export` | P→JS | YAML ファイルダウンロード |
| `raw_config` | P→JS | デバッグ用 config 表示 |
| `raw_config_error` | P→JS | config 取得エラー |

## 4. データフロー

### 4.1 Config ライフサイクル

```
initialize_config()          ← Adapter: スキーマデフォルトからシード
    ↓
apply_config_patch(ops)      ← Adapter: UI 編集（set/unset/merge）
    ↓
canonicalize_config()        ← Adapter: デフォルトとマージ
    ↓
apply_task_params()          ← Adapter: タスク依存オーバーライド
    ↓
build_config()               ← Service: df_info（data/features/split）とマージ
    ↓
prepare_run_config()         ← Adapter: 実行用最終準備（widget-only フィールド除去、Tune オーバーライド）
```

### 4.2 Fit 実行フロー

```
1. UI: sendAction("fit")
2. Widget: _on_action() → _handle_fit() → _run_job("fit")
3. Widget: Service.prepare_run_config() で完全 config 構築
4. Widget: Service.validate_config() でバリデーション
5. Widget: daemon Thread で _job_worker() 起動
6. Worker: Service.fit(config, on_progress) 呼出
7. Service: Adapter.create_model(config, df) → LizyML Model
8. Service: Adapter.fit(model, on_progress)
9. Adapter: _run_with_cancel_polling() で daemon thread + poll ループ
10. Backend thread: model.fit() 実行
11. 完了: FitSummary → fit_summary traitlet → JS 同期
```

### 4.3 Config パッチパターン（JS → Python）

```
UI 変更 → localConfig 更新 → 300ms debounce → diffToPatchOps()
→ sendAction("patch_config", {ops}) → Python apply_config_patch()
→ config traitlet 更新 → JS useTraitlet 再レンダリング
```

## 5. UI コンポーネント構成

### 5.1 コンポーネントツリー

```
App (126 LOC)
├─ Header — バックエンド情報 + ステータスバッジ
├─ Tab bar — Data | Model | Results
└─ Tab content
   ├─ DataTab (279 LOC)
   │  ├─ Target/Task セレクタ
   │  ├─ ColumnTable — 列設定 CSS Grid
   │  └─ CV 設定フォーム
   │
   ├─ ConfigTab (208 LOC) ← Fit/Tune サブタブコーディネータ
   │  ├─ FitSubTab (380 LOC)
   │  │  ├─ ModelSection
   │  │  │  ├─ TypedParamsEditor
   │  │  │  ├─ FeatureWeightsEditor
   │  │  │  └─ AdditionalParamsEditor
   │  │  ├─ DynForm (generic セクション)
   │  │  ├─ Calibration フォーム
   │  │  └─ Training フォーム
   │  ├─ ConfigFooter (65 LOC) — Import/Export YAML + Raw Config
   │  └─ TuneSubTab (160 LOC)
   │     ├─ Tuning Settings (n_trials)
   │     ├─ SearchSpace (633 LOC)
   │     ├─ Evaluation (metric 選択)
   │     └─ ConfigFooter
   │
   └─ ResultsTab (357 LOC)
      ├─ ProgressView — 進捗バー + fold 詳細
      ├─ ScoreTable — IS/OOS/OOS Std
      ├─ PlotViewer — Plotly CDN 遅延ロード
      ├─ ParamsTable — パラメータ表示
      └─ PredTable — ページネーション付き推論結果
```

### 5.2 Hooks

| Hook | 用途 |
|------|------|
| `useTraitlet(model, name)` | traitlet のリアクティブバインディング |
| `useSendAction(model)` | `action` traitlet への書込（`_ts` で一意性保証） |
| `useCustomMsg(model, handler)` | `msg:custom` リスナー |
| `usePlot(model)` | プロットキャッシュ + リクエスト管理 |

## 6. 共通型（types.py）

```python
@dataclass
class BackendInfo:
    name: str           # "lizyml"
    version: str        # "0.x.x"

@dataclass
class BackendContract:
    schema_version: int
    config_schema: dict  # JSON Schema
    ui_schema: dict      # UI セクション定義
    capabilities: dict   # 機能フラグ

@dataclass
class ConfigPatchOp:
    op: str              # "set" | "unset" | "merge"
    path: str            # "model.params.learning_rate"
    value: Any | None

@dataclass
class FitSummary:
    metrics: dict        # {metric: {is, oos, oos_std}}
    fold_count: int
    params: list[dict]

@dataclass
class TuningSummary:
    best_params: dict
    best_score: float
    trials: list[dict]
    metric_name: str
    direction: str       # "minimize" | "maximize"

@dataclass
class PredictionSummary:
    predictions: pd.DataFrame
    warnings: list[str]

@dataclass
class PlotData:
    plotly_json: str     # fig.to_json()
```

## 7. BackendAdapter Protocol

```python
class BackendAdapter(Protocol):
    # メタ
    @property
    def info(self) -> BackendInfo: ...
    def get_config_schema(self) -> dict: ...
    def get_backend_contract(self) -> BackendContract: ...

    # Config ライフサイクル
    def initialize_config(self, task=None) -> dict: ...
    def apply_config_patch(self, config, ops, task=None) -> dict: ...
    def canonicalize_config(self, config, task=None) -> dict: ...
    def apply_task_defaults(self, config, task) -> dict: ...
    def prepare_run_config(self, config, job_type, task=None) -> dict: ...
    def validate_config(self, config) -> list[dict]: ...

    # モデル操作
    def create_model(self, config, dataframe) -> Any: ...
    def fit(self, model, params, on_progress) -> FitSummary: ...
    def tune(self, model, on_progress) -> TuningSummary: ...
    def predict(self, model, data, return_shap) -> PredictionSummary: ...

    # 結果取得
    def evaluate_table(self, model) -> list[dict]: ...
    def split_summary(self, model) -> list[dict]: ...
    def importance(self, model, kind) -> dict[str, float]: ...
    def plot(self, model, plot_type) -> PlotData: ...
    def available_plots(self, model) -> list[str]: ...

    # 永続化
    def export_model(self, model, path) -> str: ...
    def load_model(self, path) -> Any: ...
    def model_info(self, model) -> dict: ...

    # 分類・推論プロット（P-013, P-015）
    def classify_best_params(self, params) -> tuple[dict, dict, dict]: ...
    def plot_inference(self, predictions, plot_type) -> PlotData: ...
```

## 8. スレッディングモデル

### 8.1 Widget レベル

```
Main Thread (Jupyter Kernel)
    │
    ├─ _run_job() → spawns:
    │   └─ daemon Thread: _job_worker()
    │       ├─ Service.fit/tune() 呼出
    │       └─ daemon Thread: elapsed timer (1s 間隔)
    │
    └─ _cancel_flag.set() → on_progress() で InterruptedError 発生
```

### 8.2 Adapter レベル（cancel-polling パターン）

```
_run_with_cancel_polling():
    ├─ daemon Thread: backend operation (model.fit/tune)
    └─ Main poll loop: on_progress() を 0.5s 間隔でチェック
        └─ InterruptedError → daemon を放棄
```

### 8.3 スレッドセーフキャッシュ

- `adapter_schema.py`: `_schema_cache` + `_schema_lock`（double-checked locking）
- `adapter_params.py`: `_eval_metrics_cache` + `_eval_metrics_lock`（同上）

## 9. Python 公開 API

```python
# 初期化
w = LizyWidget(adapter=None)        # デフォルト: LizyMLAdapter

# データロード
w.load(df, target="price")          # → LizyWidget（チェーン可）
w.set_target("price")               # → LizyWidget

# Config
w.set_config({...})                 # → LizyWidget
w.get_config()                      # → dict
w.load_config("config.yaml")        # → LizyWidget
w.save_config("config.yaml")        # → None

# 学習
w.fit(timeout=300)                  # → LizyWidget（ブロッキング）
w.tune(timeout=600)                 # → LizyWidget（ブロッキング）

# 結果
w.get_fit_summary()                 # → FitSummary | None
w.get_tune_summary()                # → TuningSummary | None
w.get_model()                       # → Any
w.predict(df, return_shap=False)    # → PredictionSummary
w.save_model("model.pkl")           # → str

# 推論
w.load_inference(df)                # → LizyWidget

# プロパティ
w.task                              # → str | None
w.cv_method                         # → str
w.cv_n_splits                       # → int
w.df_shape                          # → list[int]
w.df_columns                        # → list[dict]
```

## 10. エラーハンドリング

### エラーコード体系

| コード | 発生元 | 説明 |
|--------|--------|------|
| `NO_DATA` | Widget | データ未ロード |
| `NO_TARGET` | Widget | ターゲット未設定 |
| `TARGET_ERROR` | Widget | ターゲット設定失敗 |
| `TASK_ERROR` | Widget | タスク設定失敗 |
| `COLUMN_ERROR` | Widget | 列設定失敗 |
| `CV_ERROR` | Widget | CV 設定失敗 |
| `INVALID_PATCH` | Widget | 不正なパッチ操作 |
| `PATCH_ERROR` | Widget | パッチ適用失敗 |
| `CONFIG_ERROR` | Widget | Config 構築失敗 |
| `VALIDATION_ERROR` | Widget | Config バリデーション失敗 |
| `CANCELLED` | Widget | ジョブキャンセル |
| `BACKEND_ERROR` | Widget | ML バックエンドエラー |
| `INTERNAL_ERROR` | Widget | 内部エラー |
| `INFERENCE_ERROR` | Widget | 推論失敗 |
| `APPLY_ERROR` | Widget | ベストパラメータ適用失敗 |
| `IMPORT_ERROR` | Widget | YAML インポート失敗 |
| `EXPORT_ERROR` | Widget | YAML エクスポート失敗 |

## 11. ファイル構成

### Python（src/lizyml_widget/）

| ファイル | LOC | 責務 |
|---------|-----|------|
| `__init__.py` | 10 | パッケージエントリ |
| `_version.py` | 34 | バージョン（hatch-vcs） |
| `types.py` | 70 | 共通データ型 |
| `adapter_params.py` | 194 | タスク別パラメータ・メトリクスマッピング |
| `adapter_schema.py` | 363 | スキーマバリデーション・config 準備 |
| `adapter_contract.py` | 369 | UI Schema・BackendContract ビルダー |
| `service.py` | ~640 | WidgetService — 状態管理・アダプタ委譲・apply_best_params |
| `widget.py` | ~680 | LizyWidget — traitlets・Action・スレッド |
| `adapter.py` | 769 | BackendAdapter Protocol + LizyMLAdapter |

### TypeScript（js/src/）

| パス | LOC | 責務 |
|------|-----|------|
| `index.tsx` | 11 | anywidget ESM エントリ |
| `App.tsx` | 126 | Tab ルータ・全 traitlet バインディング |
| `tabs/DataTab.tsx` | 279 | DataFrame・CV 設定 |
| `tabs/ConfigTab.tsx` | 208 | Model Config コーディネータ（共有状態管理） |
| `tabs/FitSubTab.tsx` | 380 | Fit サブタブ（Model/Evaluation/Calibration/Training） |
| `tabs/TuneSubTab.tsx` | 160 | Tune サブタブ（Search Space/Evaluation） |
| `tabs/configHelpers.ts` | 87 | スキーマ解決・config diff ユーティリティ |
| `tabs/ResultsTab.tsx` | 357 | 結果表示・推論 |
| `hooks/useModel.ts` | 48 | traitlet フック |
| `hooks/usePlot.ts` | 57 | プロットキャッシュ |
| `components/DynForm.tsx` | 383 | JSON Schema → フォーム生成 |
| `components/SearchSpace.tsx` | 633 | Tune search space UI |
| `components/ModelEditors.tsx` | 460 | モデルパラメータエディタ |
| `components/ConfigFooter.tsx` | 65 | Import/Export YAML + Raw Config モーダル |
| `components/Header.tsx` | 31 | ヘッダー |
| `components/Accordion.tsx` | 41 | 折畳セクション |
| `components/NumericStepper.tsx` | 59 | 数値入力 |
| `components/ColumnTable.tsx` | 79 | 列設定テーブル |
| `components/ProgressView.tsx` | 98 | 進捗表示 |
| `components/PlotViewer.tsx` | 71 | Plotly レンダラ |
| `components/ScoreTable.tsx` | 64 | メトリクステーブル |
| `components/ParamsTable.tsx` | 45 | パラメータテーブル |
| `components/PredTable.tsx` | 105 | 推論結果テーブル |

### テスト（tests/）

| ファイル | LOC | 対象 |
|---------|-----|------|
| `test_adapter_core.py` | 708 | Adapter コア（fit/tune/predict/plot） |
| `test_adapter_config.py` | 730 | Config ライフサイクル |
| `test_adapter_schema.py` | 430 | スキーマ正規化・strip |
| `test_adapter_tune.py` | 797 | Tune/search space |
| `test_widget_api.py` | 1,665 | Widget 公開 API |
| `test_service.py` | 716 | Service 状態管理 |
| `test_e2e.py` | 408 | E2E ワークフロー |
| `test_frontend_contract.py` | 305 | BackendContract スキーマ |

## 12. 既知の乖離・技術的負債

### BLUEPRINT との乖離

| # | BLUEPRINT 記載 | 実装の実態 | 影響 |
|---|---------------|-----------|------|
| 1 | Adapter は `adapter.py` 単体 | 4ファイル構成（adapter/params/schema/contract） | BLUEPRINT §3 の構造図に未反映 |
| 2 | config traitlet は常に正規化済み | patch_config unset 後に非正規化状態の可能性 | Phase 26 残存 |
| 3 | 全入力パス（UI/API/YAML）が同一正規化 | UI patch ≠ Python API パス | Phase 26 残存 |

### 解決済み（Phase 27）

| # | 解決内容 |
|---|---------|
| 1 | Widget の adapter_params 直接依存を除去 → Service.apply_best_params() に移動 |
| 2 | calibration_methods を ui_schema に移動（JS ハードコード除去） |
| 3 | cv_strategies を BackendContract capabilities に移動（Widget ハードコード除去） |
| 4 | classify_best_params / plot_inference を Protocol に追加（duck-typing 除去） |
| 5 | ConfigTab を FitSubTab + TuneSubTab + ConfigFooter に分割（867→208行） |
| 6 | test_adapter.py を4ファイルに分割（2,759→max 797行） |
| 7 | Widget fit()/tune() を _run_blocking_job() に統一 |

### 残存するコード品質課題

| # | 問題 | ファイル | 対応 |
|---|------|---------|------|
| 1 | SearchSpace 633 LOC | `components/SearchSpace.tsx` | 複雑度高、将来的に分割検討 |
