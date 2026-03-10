## LizyML-Widget 開発計画

---

### フェーズ概要

| Phase | 名称 | 目標 | 成果物 |
|-------|------|------|--------|
| 0 | プロジェクト基盤 | ゼロから動く Widget を表示する | pyproject.toml, esbuild, anywidget Hello World |
| 1 | Backend Adapter + Service | LizyML を Python から呼べるようにする | types.py, adapter.py, service.py, テスト |
| 2 | Data タブ | DataFrame 読み込み → Target 選択 → 自動判定 | DataTab UI + traitlets 連携 |
| 3 | Config タブ (Fit) | JSON Schema からフォーム動的生成 → Fit 実行 | ConfigTab (Fit) + action 連携 |
| 4 | Results タブ (Fit) | 進捗表示 → Score → Plotly プロット → Accordion | ResultsTab + msg:custom 連携 |
| 5 | Tune | Search Space UI → Tune 実行 → Tune 結果表示 | ConfigTab (Tune) + ResultsTab 拡張 |
| 6 | Inference | load_inference() → 推論実行 → 結果表示 | Inference セクション |
| 7 | Config Import/Export + Python API | YAML I/O + プログラム API の完成 | load_config / save_config / save_model 等 |
| 8 | 品質・配布 | CI・テスト拡充・PyPI 配布 | pyproject.toml artifacts, CI, README |

各フェーズ末尾に **完了条件** を定義する。フェーズは順番に実施する（前フェーズの成果物が後フェーズの前提）。

---

### Phase 0: プロジェクト基盤

**目標:** `uv run jupyter lab` でセルに Hello World Widget が表示される状態を作る。

#### 0-1. Python プロジェクト初期化

- `pyproject.toml` 作成（hatchling ビルド、`artifacts = ["src/lizyml_widget/static/**"]`）
- 依存: `anywidget`, `traitlets`, `pandas`
- 開発依存: `pytest`, `ruff`, `mypy`, `jupyterlab`
- `uv lock` で `uv.lock` 生成
- `src/lizyml_widget/__init__.py` — パッケージエントリ（`from .widget import LizyWidget`）
- `src/lizyml_widget/widget.py` — 最小 LizyWidget（`_esm`, `_css`, ステータス traitlet 1 つ）
- `src/lizyml_widget/static/` ディレクトリ作成（.gitkeep）

#### 0-2. TypeScript / esbuild 初期化

- `js/package.json` 作成（`preact`, `esbuild`, `typescript` を依存に追加）
- `js/tsconfig.json`（strict, jsx: "react-jsx", jsxImportSource: "preact"）
- `js/src/index.tsx` — anywidget エントリ。`<div>Hello LizyML Widget</div>` を render
- `js/src/widget.css` — `.lzw-root` の最小スタイル
- `pnpm dev` / `pnpm build` スクリプトを package.json に定義
  - `dev`: `esbuild src/index.tsx --bundle --format=esm --jsx=automatic --jsx-import-source=preact --outfile=../src/lizyml_widget/static/widget.js --watch`
  - `build`: 上記 + `--minify`

#### 0-3. 開発環境の検証

- `pnpm build` → `src/lizyml_widget/static/widget.js` が生成される
- `uv run jupyter lab` → Notebook でセルに `LizyWidget()` と入力 → Hello World が表示される
- Ruff / mypy が通る

#### 0-4. Git 初期化

- `.gitignore`（`node_modules/`, `dist/`, `*.egg-info/`, `.mypy_cache/`, `src/lizyml_widget/static/widget.js`, `src/lizyml_widget/static/widget.css`）
- `develop` ブランチ作成
- CLAUDE.md / BLUEPRINT.md / PLAN.md / HISTORY.md を初期コミット

**完了条件:** Notebook セルに `LizyWidget()` を置くと「Hello LizyML Widget」が表示される。`uv run ruff check .` と `uv run mypy src/lizyml_widget/` がエラーなしで通る。

---

### Phase 1: Backend Adapter + Service

**目標:** Python 側で LizyML を呼べる基盤を作る。UI はまだ不要。

#### 1-1. 共通型の定義

- `src/lizyml_widget/types.py`
  - `BackendInfo`, `ConfigSchema`, `FitSummary`, `TuningSummary`, `PredictionSummary`, `PlotData`
  - BLUEPRINT §3.3.1 の通り

#### 1-2. BackendAdapter Protocol

- `src/lizyml_widget/adapter.py`
  - `BackendAdapter` Protocol 定義（BLUEPRINT §3.3.2）
  - `LizyMLAdapter` クラス（`BackendAdapter` の実装）
    - `info` → `BackendInfo(name="lizyml", version=lizyml.__version__)`
    - `get_config_schema()` → LizyML の Config を JSON Schema に変換
    - `validate_config()` → LizyML のバリデーション呼び出し
    - `create_model()` → LizyML のモデル生成
    - `fit()` → FitSummary への変換
    - `plot()` → PlotData への変換
    - `available_plots()` → タスクに応じたプロットタイプ一覧
    - `importance()` → dict 変換
    - `evaluate_table()` / `split_summary()` → list[dict] 変換
  - 残りのメソッド（`tune`, `predict`, `export_model`, `load_model`, `model_info`）はスタブ（Phase 5, 6, 7 で実装）

#### 1-3. WidgetService

- `src/lizyml_widget/service.py`
  - `WidgetService` クラス
  - コンストラクタ: `BackendAdapter` を受け取る
  - `load_data(df, target)` → df_info dict を生成（列情報・ユニーク数・dtype）
  - `set_target(target)` → Task 自動判定・Column 自動設定・CV デフォルト設定（BLUEPRINT §5.2 のルール）
  - `update_column(name, excluded, col_type)` → 個別カラム設定変更
  - `update_cv(strategy, n_splits, group_column)` → CV 設定変更
  - `build_config()` → df_info + ユーザー設定 → LizyML Config dict を構築
  - `fit(config, on_progress)` → Adapter.create_model → Adapter.fit → FitSummary を返す
  - 内部状態: `_df`, `_df_info`, `_model`（直近の学習済みモデル）

#### 1-4. テスト

- `tests/test_types.py` — 共通型の生成・シリアライズ
- `tests/test_service.py`
  - Task 自動判定の境界値テスト（binary / multiclass / regression）
  - Column 自動設定テスト（ID 除外・Const 除外・型推定）
  - CV デフォルト設定テスト
- `tests/test_adapter.py`
  - 小規模データで LizyMLAdapter.fit() → FitSummary の構造を検証
  - plot() → PlotData.plotly_json が有効な JSON であること
- pyproject.toml に `lizyml[plots,tuning,calibration,explain]` を依存に追加

**完了条件:** `uv run pytest` が全件パス。`WidgetService` が df_info 生成・fit 実行・FitSummary 返却まで通る。

---

### Phase 2: Data タブ

**目標:** Widget で DataFrame を読み込み、Target 選択・Column 設定・CV 設定ができる。

#### 2-1. traitlets の実装

- `widget.py` に以下の traitlets を追加:
  - `backend_info: Dict`
  - `df_info: Dict`
  - `status: Unicode` （初期値 `"idle"`）
  - `action: Dict`
- `load()` メソッドの実装:
  - DataFrame を `_service.load_data()` に渡す
  - `df_info` traitlet を更新
  - target 指定時は `_service.set_target()` も呼ぶ
  - `status` → `"data_loaded"`
- `@observe("action")` ハンドラ:
  - `set_target` → `_service.set_target()` → `df_info` 更新
  - `update_column` → `_service.update_column()` → `df_info` 更新
  - `update_cv` → `_service.update_cv()` → `df_info` 更新

#### 2-2. JS 基盤コンポーネント

- `js/src/hooks/useModel.ts`
  - anywidget model ラッパーフック（traitlet の get/set/listen を Preact state に接続）
  - `useTraitlet<T>(model, name)` → `[value, setValue]`
  - `sendAction(model, type, payload)` ヘルパー
- `js/src/App.tsx`
  - Header + Tabs 構造（BLUEPRINT §5.1）
  - タブ: Data / Config / Results。有効条件の制御
- `js/src/components/Header.tsx`
  - バックエンドバッジ + ステータスバッジ

#### 2-3. DataTab 実装

- `js/src/tabs/DataTab.tsx`
  - DataFrame 情報表示（行数 × 列数）
  - Target / Task セクション
    - Target: df_info.columns からドロップダウン生成。変更時に `set_target` action 送信
    - Task: 自動判定結果の表示。ドロップダウンで変更可能
  - Column Settings テーブル（`js/src/components/ColumnTable.tsx`）
    - Column / Uniq / Excl チェックボックス / Type ドロップダウン
    - バッジ表示（[ID], [Const]）
    - 変更時に `update_column` action 送信
  - Cross Validation セクション
    - Strategy ドロップダウン / Folds 数値入力 / Group column（条件付き表示）
    - 変更時に `update_cv` action 送信
  - Feature Summary（リアルタイム集計表示）
- `js/src/components/Accordion.tsx`
  - 汎用の展開/折りたたみコンポーネント（Data タブ・Results タブで共用）

**完了条件:** `w = LizyWidget(); w.load(df, target="y"); w` でセルに Data タブが表示される。Target 変更 → Column 自動設定が動く。Column / CV の手動変更が反映される。

---

### Phase 3: Config タブ (Fit)

**目標:** JSON Schema からフォームを動的生成し、Fit を実行して結果を Python に格納できる。

#### 3-1. Config traitlets + Service 拡張

- `widget.py` に追加:
  - `config_schema: Dict`
  - `config: Dict`
  - `job_type: Unicode`
  - `job_index: Int`
  - `progress: Dict`
  - `elapsed_sec: Float`
  - `fit_summary: Dict`
  - `error: Dict`
- `load()` 時に `config_schema` を `adapter.get_config_schema()` から設定
- Action ハンドラ追加:
  - `update_config` → config dict の更新 + `adapter.validate_config()` でバリデーション
  - `fit` → `threading.Thread` でバックグラウンド実行（BLUEPRINT §3.7 のパターン）
    - `status` → `"running"`、`job_type` → `"fit"`、`job_index` インクリメント
    - `on_progress` コールバックで `progress` + `elapsed_sec` を更新
    - 完了時: `fit_summary` 設定、`status` → `"completed"`
    - 失敗時: `error` 設定、`status` → `"failed"`
  - `cancel` → 実行中スレッドへのキャンセルフラグ設定

#### 3-2. DynForm コンポーネント

- `js/src/components/DynForm.tsx`
  - `config_schema` (JSON Schema) を受け取り、セクションごとにフォームを動的生成
  - 型マッピング（BLUEPRINT §5.3）:
    - `number`/`integer` → NumberInput
    - `boolean` → Switch
    - `string` + `enum` → Select
    - `string` → TextInput
    - `array` → タグ入力
  - `default` → 初期値、`description` → ツールチップ
  - 変更をデバウンス（300ms）して `update_config` action を送信

#### 3-3. ConfigTab (Fit サブタブ) 実装

- `js/src/tabs/ConfigTab.tsx`
  - Fit / Tune サブタブ切り替え（Phase 3 では Fit のみ実装、Tune は disabled）
  - Fit サブタブ:
    - Model セクション（DynForm でモデル選択 + ハイパーパラメータ）
    - Training セクション（DynForm）
    - Evaluation セクション（DynForm）
    - Calibration セクション（binary 時のみ表示、DynForm）
  - Fit ボタン（有効条件: status が data_loaded 以降 + Model 選択済み）
    - クリック → `fit` action 送信

**完了条件:** Config タブでモデル選択 → パラメータ編集 → Fit ボタン押下 → Python 側で fit が走り `fit_summary` traitlet に結果が格納される。バリデーションエラーがフォームに表示される。

---

### Phase 4: Results タブ (Fit)

**目標:** Fit の進捗 → Score テーブル → Plotly プロット → Accordion を表示する。

#### 4-1. ProgressView

- `js/src/components/ProgressView.tsx`
  - プログレスバー（`progress.current` / `progress.total`）
  - 経過時間（`elapsed_sec`）
  - Fold ログ（進捗メッセージのリスト表示）
  - Cancel ボタン → `cancel` action 送信

#### 4-2. ScoreTable

- `js/src/components/ScoreTable.tsx`
  - `fit_summary.metrics` から IS / OOS / OOS Std のテーブルを生成
  - OOS Std は CV 時のみ列を表示

#### 4-3. PlotViewer + Plotly 連携

- `js/src/hooks/usePlot.ts`
  - `request_plot` action 送信 + `msg:custom` 受信のフック（BLUEPRINT §7.4）
  - プロットデータのキャッシュ管理
- `js/src/components/PlotViewer.tsx`
  - `available_plots` からセレクタを生成
  - セレクタ変更 → `requestPlot()` → Plotly.js を CDN からロードして描画
  - `window.Plotly` フォールバック付き
- Widget 側:
  - `request_plot` action → `adapter.plot()` → `widget.send({"type": "plot_data", ...})`
  - `available_plots` traitlet を fit 完了時に設定

#### 4-4. ResultsTab 実装

- `js/src/tabs/ResultsTab.tsx`
  - 状態に応じた表示切り替え:
    - `status == "idle"` or `"data_loaded"`: ガイドテキスト
    - `status == "running"`: ProgressView
    - `status == "completed"`: Score → Learning Curve → Plots → Accordion
    - `status == "failed"`: エラー表示 + [Show Full Traceback] + [Re-run]
  - ヘッダー: `Fit #3 ── LightGBM ── ✓ Completed`
  - Accordion セクション:
    - Feature Importance（Plotly 棒グラフ）
    - Fold Details テーブル（CV 時のみ）— `js/src/components/ParamsTable.tsx` を共用
    - Parameters テーブル
- `js/src/components/ParamsTable.tsx`
  - key-value テーブルの汎用コンポーネント

#### 4-5. 自動タブ切り替え

- Fit 開始時 → Results タブに自動切り替え
- App.tsx のタブ状態を status の変化に連動

**完了条件:** Fit 実行 → Results タブで進捗バーがリアルタイム更新 → 完了後に Score テーブル・Learning Curve・Plotly プロット・Feature Importance・Fold Details・Parameters が表示される。エラー時はエラー画面が表示される。

---

### Phase 5: Tune

**目標:** Tune の Search Space UI → Tune 実行 → Tune 結果の表示を完成させる。

#### 5-1. Adapter + Service 拡張 (Tune)

- `LizyMLAdapter.tune()` を実装（Phase 1-2 のスタブを完成）
  - `on_progress` コールバックで Trial 進捗を返す
  - `TuningSummary` への変換
- `WidgetService.tune(config, on_progress)` を追加
- `widget.py`:
  - `tune_summary: Dict` traitlet 追加（Phase 3 で定義済みなら確認）
  - `tune` action ハンドラ実装（fit と同じスレッドパターン）

#### 5-2. SearchSpace コンポーネント

- `js/src/components/SearchSpace.tsx`
  - `config_schema` からパラメータ一覧を生成
  - 各パラメータ行: Param 名 / Mode ドロップダウン / Config 値
  - Mode 選択肢（BLUEPRINT §5.3）:
    - `float`/`integer` → Fixed / Range
    - `enum`/`string`/`boolean` → Fixed / Choice
  - Mode = Range: min / max / distribution / step（integer のみ）
  - Mode = Choice: チップ形式の複数選択
  - Mode = Fixed: 単一値入力

#### 5-3. ConfigTab (Tune サブタブ) 完成

- ConfigTab の Tune サブタブを有効化
  - Model セクション（Fit と独立したモデル選択）
  - Settings セクション（n_trials / timeout / scoring）
  - Search Space セクション（SearchSpace コンポーネント）
  - Tune ボタン（有効条件: data_loaded + Model 選択 + Range/Choice が 1 つ以上）

#### 5-4. ResultsTab (Tune) 拡張

- Tune 実行中: ProgressView を Trial 進捗形式に拡張（`Trial 12 / 100  Best AUC = 0.891`）
- Tune 完了: Fit 完了と同じ構成 + 先頭に追加セクション:
  - Optimization History（Plotly 収束プロット）
  - Best Params テーブル + [Apply to Fit] ボタン
  - Trial Results（Accordion 内）

**完了条件:** Tune タブで Search Space 設定 → Tune 実行 → Trial 進捗リアルタイム表示 → 完了後に Optimization History・Best Params・Score・Plots が表示される。[Apply to Fit] で Fit タブにパラメータがコピーされる。

---

### Phase 6: Inference

**目標:** 学習済みモデルに新データを渡して推論し、結果を Widget 内に表示する。

#### 6-1. Adapter + Service 拡張 (Inference)

- `LizyMLAdapter.predict()` を実装（Phase 1-2 のスタブを完成）
  - `return_shap` 対応
  - `PredictionSummary` への変換
- `WidgetService.predict(data, return_shap)` を追加
- `widget.py`:
  - `inference_result: Dict` traitlet
  - `load_inference(df)` Python API メソッド — 推論用 DataFrame を内部に保持
  - `run_inference` action ハンドラ — `_service.predict()` → 結果を traitlet に格納

#### 6-2. Inference UI

- ResultsTab 末尾に Inference セクションを追加:
  - `load_inference()` 未呼び出し時: ガイドテキスト
  - 呼び出し済み: [Return SHAP values] チェック + [Run Inference] ボタン
  - 推論完了後:
    - `js/src/components/PredTable.tsx` — 予測結果テーブル（ページネーション 50 行/ページ）
    - [Download CSV] — ブラウザから CSV ダウンロード
    - Prediction Distribution（Plotly ヒストグラム）
    - SHAP Summary（有効時のみ）
    - Warnings（ある場合のみ）

**完了条件:** `w.load_inference(test_df)` → Widget の Inference セクションで [Run Inference] → 予測テーブル・分布ヒストグラムが表示される。CSV ダウンロードが動作する。

---

### Phase 7: Config Import/Export + Python API

**目標:** YAML の入出力とプログラム用 Python API を完成させる。

#### 7-1. Config Import/Export

- Python 側:
  - `load_config(path)` — YAML/JSON ファイルを読み込み、config traitlet にセット。data/features/split は df_info にも反映
  - `save_config(path)` — 現在の config を YAML ファイルに書き出す
  - `set_config(config)` — dict を直接セット
- JS 側:
  - [Import YAML] ボタン — `<input type="file">` でファイルを選択、内容を action で Python に送信
  - [Export YAML] ボタン — action → Python が YAML 生成 → msg:custom でブラウザにダウンロード
  - [Raw Config] ボタン — 現在の config を YAML テキストでモーダル表示

#### 7-2. Python API 完成

- `get_fit_summary()` → `FitSummary | None`
- `get_tune_summary()` → `TuningSummary | None`
- `get_model()` → 内部モデルオブジェクト（バックエンド依存型）
- `predict(df, *, return_shap=False)` → `PredictionSummary`（UI を経由しないプログラム推論）
- `save_model(path)` → `adapter.export_model()` 経由

#### 7-3. Adapter 残りメソッド

- `LizyMLAdapter.export_model()` / `load_model()` / `model_info()` を実装

#### 7-4. テスト追加

- `tests/test_widget_api.py`
  - `load()` → `set_config()` → `get_fit_summary()` の統合テスト
  - `load_config()` / `save_config()` の YAML 読み書きテスト
  - `predict()` のプログラム実行テスト

**完了条件:** `w.load(df).load_config("config.yaml")` → Fit → `w.get_fit_summary()` が FitSummary を返す。`w.save_config("out.yaml")` が正しい YAML を書き出す。`w.save_model("./model")` でモデルが保存される。

---

### Phase 8: 品質・配布

**目標:** CI 整備・テスト拡充・PyPI 配布可能な状態にする。

#### 8-1. テスト拡充

- Adapter 全メソッドの統合テスト
- Service 層の境界値テスト拡充
- Widget traitlets の状態遷移テスト（idle → data_loaded → running → completed/failed）
- エラーハンドリングテスト（不正な DataFrame / 不正な Config / バックエンドエラー）

#### 8-2. CI (GitHub Actions)

- main への PR 時のみ実行
- マトリクス: Python 3.10 / 3.11 / 3.12
- ステップ:
  1. `uv sync`
  2. `cd js && pnpm install && pnpm build`
  3. `uv run ruff check .`
  4. `uv run ruff format --check .`
  5. `uv run mypy src/lizyml_widget/`
  6. `uv run pytest`

#### 8-3. PyPI 配布準備

- `pyproject.toml` に metadata 追加（author, license, description, classifiers, urls）
- README.md 作成（インストール手順・クイックスタート・スクリーンショット）
- `hatchling` の `artifacts` 設定確認（`src/lizyml_widget/static/**`）
- `uv build` → wheel に static/ が含まれることを確認
- PyPI テスト配布 → `pip install lizyml-widget` で Notebook から動作確認

#### 8-4. 環境別動作確認

- Jupyter Notebook
- JupyterLab
- Google Colab
- VS Code Notebooks

**完了条件:** `pip install lizyml-widget` → Jupyter / Colab / VS Code で Widget の全機能が動作する。CI が main PR で自動実行される。

---

### 依存関係グラフ

```
Phase 0 (基盤)
  │
  ▼
Phase 1 (Adapter + Service)
  │
  ├─────────────────────┐
  ▼                     ▼
Phase 2 (Data タブ)    (テスト基盤)
  │
  ▼
Phase 3 (Config/Fit)
  │
  ▼
Phase 4 (Results/Fit)
  │
  ├──────────┬──────────┐
  ▼          ▼          ▼
Phase 5    Phase 6    Phase 7
(Tune)     (Inference) (Import/Export + API)
  │          │          │
  └──────────┴──────────┘
             │
             ▼
          Phase 8 (品質・配布)
```

Phase 5 / 6 / 7 は互いに独立しており並行実施可能。
