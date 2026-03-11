## LizyML-Widget 開発計画

---

### フェーズ概要

| Phase | 名称 | 目標 | ステータス |
|-------|------|------|-----------|
| 0 | プロジェクト基盤 | ゼロから動く Widget を表示する | ✅ 完了 |
| 1 | Backend Adapter + Service | LizyML を Python から呼べるようにする | ✅ 完了 |
| 2 | Data タブ | DataFrame 読み込み → Target 選択 → 自動判定 | ✅ 完了 |
| 3 | Model タブ (Fit) | JSON Schema からフォーム動的生成 → Fit 実行 | ✅ 完了 |
| 4 | Results タブ (Fit) | 進捗表示 → Score → Plotly プロット → Accordion | ✅ 完了 |
| 5 | Tune | Search Space UI → Tune 実行 → Tune 結果表示 | ✅ 完了 |
| 6 | Inference | load_inference() → 推論実行 → 結果表示 | ⚠️ ほぼ完了（Inference plot 未対応） |
| 7 | Config Import/Export + Python API | YAML I/O + プログラム API の完成 | ⚠️ ほぼ完了（load_model/model_info 未実装） |
| 8 | 品質・配布 | CI・テスト拡充・PyPI 配布 | 🔲 未着手（CI / README / PyPI） |
| 9 | BLUEPRINT 乖離解消 | 要件監査是正 | ✅ 完了 |
| 10 | 実画面確認ベース残開発 | Notebook 実画面監査の乖離解消 | ⚠️ ほぼ完了（一部残作業あり） |
| 11 | 仕様同期 + 残 plot 実装 | 監査で発見した乖離の解消 | 🔲 未着手 |
| 12 | UI ユーザビリティ改善 | マウス操作性向上・Calibration バグ修正・SearchSpace Choice 拡充 | 🔲 未着手 |

各フェーズ末尾に **完了条件** を定義する。フェーズは順番に実施する（前フェーズの成果物が後フェーズの前提）。

---

### Phase 0: プロジェクト基盤 ✅

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

### Phase 1: Backend Adapter + Service ✅

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

### Phase 2: Data タブ ✅

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
  - タブ: Data / Model / Results。有効条件の制御
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

### Phase 3: Model タブ (Fit) ✅

**目標:** JSON Schema からフォームを動的生成し、Fit を実行して結果を Python に格納できる。

#### 3-1. Model traitlets + Service 拡張

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

#### 3-3. Model タブ（ConfigTab / Fit サブタブ）実装

- `js/src/tabs/ConfigTab.tsx`
  - Fit / Tune サブタブ切り替え（Phase 3 では Fit のみ実装、Tune は disabled）
  - Fit サブタブ:
    - Model セクション（DynForm でモデル選択 + ハイパーパラメータ）
    - Training セクション（DynForm）
    - Evaluation セクション（DynForm）
    - Calibration セクション（binary 時のみ表示、DynForm）
  - Fit ボタン（有効条件: status が data_loaded 以降 + Model 選択済み）
    - クリック → `fit` action 送信

**完了条件:** Model タブでモデル選択 → パラメータ編集 → Fit ボタン押下 → Python 側で fit が走り `fit_summary` traitlet に結果が格納される。バリデーションエラーがフォームに表示される。

---

### Phase 4: Results タブ (Fit) ✅

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

### Phase 5: Tune ✅

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

#### 5-3. Model タブ（ConfigTab / Tune サブタブ）完成

- Model タブ（ConfigTab）の Tune サブタブを有効化
  - Model セクション（Fit と独立したモデル選択）
  - Settings セクション（n_trials / metric）
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

### Phase 6: Inference ⚠️

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

- ✅ ResultsTab 末尾に Inference セクションを追加
- ✅ `load_inference()` 未呼び出し時: ガイドテキスト
- ✅ 呼び出し済み: [Return SHAP values] チェック + [Run Inference] ボタン
- ✅ 推論完了後:
  - ✅ `js/src/components/PredTable.tsx` — 予測結果テーブル（ページネーション 50 行/ページ）
  - ✅ [Download CSV] — ブラウザから CSV ダウンロード
  - ❌ Prediction Distribution（Plotly ヒストグラム）— UI はあるが adapter.py に `prediction-distribution` plot type 未実装
  - ❌ SHAP Summary（有効時のみ）— UI はあるが adapter.py に `shap-summary` plot type 未実装
  - ✅ Warnings（ある場合のみ）

**完了条件:** `w.load_inference(test_df)` → Widget の Inference セクションで [Run Inference] → 予測テーブル・分布ヒストグラムが表示される。CSV ダウンロードが動作する。

---

### Phase 7: Config Import/Export + Python API ⚠️

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

- ✅ `LizyMLAdapter.export_model()` — 実装済み
- ❌ `LizyMLAdapter.load_model()` — `NotImplementedError`（将来対応）
- ❌ `LizyMLAdapter.model_info()` — `NotImplementedError`（将来対応）

#### 7-4. テスト追加

- `tests/test_widget_api.py`
  - `load()` → `set_config()` → `get_fit_summary()` の統合テスト
  - `load_config()` / `save_config()` の YAML 読み書きテスト
  - `predict()` のプログラム実行テスト

**完了条件:** `w.load(df).load_config("config.yaml")` → Fit → `w.get_fit_summary()` が FitSummary を返す。`w.save_config("out.yaml")` が正しい YAML を書き出す。`w.save_model("./model")` でモデルが保存される。

---

### Phase 8: 品質・配布 🔲

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
Phase 0 (基盤) ✅
  │
  ▼
Phase 1 (Adapter + Service) ✅
  │
  ├─────────────────────┐
  ▼                     ▼
Phase 2 (Data タブ) ✅  (テスト基盤) ✅
  │
  ▼
Phase 3 (Model/Fit) ✅
  │
  ▼
Phase 4 (Results/Fit) ✅
  │
  ├──────────┬──────────┐
  ▼          ▼          ▼
Phase 5 ✅ Phase 6 ⚠️ Phase 7 ⚠️
(Tune)     (Inference) (Import/Export + API)
  │          │          │
  └──────────┴──────────┘
             │
  ┌──────────┼──────────┐
  ▼          ▼          ▼
Phase 9 ✅ Phase 10 ⚠️ Phase 11 🔲
(乖離解消) (実画面修正) (仕様同期+残plot)
             │          │
             └──────────┘
                  │
                  ▼
             Phase 8 🔲 (品質・配布)
```

Phase 11 は Phase 10 の残作業 + 監査乖離の解消。Phase 8 は全フェーズ完了後に実施。

---

### Phase 9: BLUEPRINT乖離解消（要件監査是正） ✅

**目標:** 現在実装を BLUEPRINT（特に §3.4-§3.6 / §4.1 / §5.1-§5.4）に一致させ、Data/Model 設定の不整合で Fit/Tune が失敗する状態を解消する。

#### 9-1. Data タブ設定キー整合（必須）

- `src/lizyml_widget/service.py`
  - `build_config()` を BLUEPRINT §5.2 の保存キーに合わせる
    - `task` は `data.task` ではなくトップレベル `task` へ出力
    - `data.group_col` / `data.time_col` を `df_info.cv` から構築
    - `split` に `method`, `n_splits`, `random_state`, `shuffle`, `gap`, `purge_gap`, `embargo`, `train_size_max`, `test_size_max` を条件付きで出力
  - `load_data()` / `update_cv()` の `df_info.cv` 構造を拡張（上記キーを保持）
- `js/src/tabs/DataTab.tsx`
  - CV Strategy の選択肢を `kfold` / `stratified_kfold` / `group_kfold` / `time_series` / `purged_time_series` / `group_time_series` に更新
  - Strategy 依存フィールドを追加
    - `Random state`, `Shuffle`, `Group column`, `Time column`, `Gap`, `Purge gap`, `Embargo`, `Train size max`, `Test size max`
  - `update_cv` payload を拡張し、上記設定を欠落なく送信
- `src/lizyml_widget/widget.py`
  - `_handle_update_cv()` で新規 CV payload を受理し service に伝搬

#### 9-2. Model タブ設定キー・初期値整合（LightGBM + Tuning）

- `js/src/App.tsx`
  - タブ表示名を `Config` から `Model` に変更（実装コンポーネント名 `ConfigTab` は維持）
- `js/src/tabs/ConfigTab.tsx`
  - Fit サブタブの設定対象を BLUEPRINT §5.3 に一致
    - `config_version=1`（読み取り専用）
    - `model.name="lgbm"`、`model.params`、`model.auto_num_leaves=true`、`model.num_leaves_ratio=1.0`、`model.min_data_in_leaf_ratio=0.01`、`model.min_data_in_bin_ratio=0.01`、`model.feature_weights=null`、`model.balanced=null`
    - `training.seed=42`、`training.early_stopping.enabled=true`、`training.early_stopping.rounds=150`、`training.early_stopping.validation_ratio=0.1`、`training.early_stopping.inner_valid=null`
    - `evaluation.metrics=[]`
    - `calibration=null`（OFF）。ON 時は `method="platt"`, `n_splits=5`, `params={}` を編集可能
    - `output_dir=null`
  - Tune サブタブのキーを `tuning.optuna.params.*` / `tuning.optuna.space` に変更
    - `n_trials=50`, `space={}`（direction は metric に応じて自動決定、timeout は未使用のため削除）
  - `SearchSpace` の編集結果を `localConfig.tuning.optuna.space` に統合
- `js/src/components/SearchSpace.tsx`
  - `Fixed/Range/Choice` の値を `tuning.optuna.space` に直結するデータ形へ変換
  - `range` の `step`（integer 時）と `log` の扱いを仕様化

#### 9-3. 初期 Config 生成とバリデーション導線

- `src/lizyml_widget/widget.py`
  - `load()` 時に `config_schema` の default を展開して `config` 初期値を生成（未編集でも実行可能にする）
  - `_handle_update_config()` で `service.validate_config(full_config)` を実行し、検証結果を `error` へ反映
  - `fit` / `tune` 実行前にも最終バリデーションを実施し、`VALIDATION_ERROR` を返却
- `src/lizyml_widget/service.py`
  - `build_config()` の戻り値が `adapter.validate_config()` を常に通ることを前提とした正規化処理を追加

#### 9-4. Results タブ・Action・Plot 命名整合

- `src/lizyml_widget/adapter.py`
  - plot type 命名を UI/BLUEPRINT と統一
    - `importance` -> `feature-importance`
    - `tuning` -> `optimization-history`
  - `available_plots()` の返却値も同じ命名へ変更
- `js/src/tabs/ResultsTab.tsx`
  - 上記 plot type 名を前提に表示ロジックを修正
  - Tune 結果の [Apply to Fit] を有効化した場合、`apply_best_params` action を送信し、Model(Fit)へ反映・タブ切替
- `src/lizyml_widget/widget.py`
  - `apply_best_params` action handler を追加
  - `request_inference_plot` action handler を追加（BLUEPRINT §3.6）

#### 9-5. 非同期進捗とキャンセルの実装完成

- `src/lizyml_widget/widget.py`
  - `fit` / `tune` 実行で `on_progress` を service 経由で adapter に渡し `progress` を更新
  - `_cancel_flag` を worker 内で監視し `CANCELLED` 遷移を実装
- `src/lizyml_widget/adapter.py`
  - backend 側 progress callback 連携点を実装（利用可能な API に合わせる）
- `js/src/components/ProgressView.tsx`
  - Fold/Trial 表示フォーマットを `job_type` に応じて出し分け

#### 9-6. Python API 契約準拠（BLUEPRINT §4.1）

- `src/lizyml_widget/widget.py`
  - `load()` / `set_config()` / `load_config()` の戻り値を `self` に変更（チェーン可能）
  - `get_fit_summary()` / `get_tune_summary()` の戻り値型を `dict` 固定から `None` 許容へ変更（空時 `None`）
  - `predict()` の戻り値を `PredictionSummary` に変更
  - `save_model(path)` を追加し `adapter.export_model()` を呼び出す
- `src/lizyml_widget/adapter.py`
  - `export_model()` / `load_model()` / `model_info()` の実装

#### 9-7. Import/Export UI 実装（現在 placeholder）

- `js/src/tabs/ConfigTab.tsx`
  - `Import YAML` / `Export YAML` / `Raw Config` ボタンの disabled を解除し action 連携
- `src/lizyml_widget/widget.py`
  - import/export 用 action handler を追加し、YAML 文字列を `msg:custom` 経由で送受信
  - 読み込み時に `data` / `features` / `split` / `task` を Data タブへ反映

#### 9-8. テスト拡張（乖離再発防止）

- `tests/test_service.py`
  - `build_config()` が `task` top-level・`data.group_col`・時系列 split パラメータを正しく出力するテストを追加
  - `adapter.validate_config()` が通る構成を回帰テスト化
- `tests/test_widget_api.py`
  - API チェーン（`load().set_config().load_config()`）と `save_model()` のテストを追加
  - `apply_best_params` / `request_inference_plot` action dispatch テストを追加
- `tests/test_adapter.py`
  - plot type 命名変更（`feature-importance`, `optimization-history`）のテストを追加
- Notebook 統合テスト
  - `w.load(...); fit` が `status=completed` まで到達する E2E を追加

#### 9-9. ツールチェーン要件整合

- `js/package.json`
  - `lint` スクリプトを追加（`eslint` + `typescript-eslint` 導入）
- CI/ローカルコマンド
  - `cd js && pnpm lint` を品質ゲートに追加し AGENTS.md のコマンド仕様に一致させる

**完了条件:**

- 既定設定（ユーザー未編集）で `w.load(df, target="y")` → Fit が `status="completed"` になる
- Data/Model タブの全設定項目が BLUEPRINT §5.2/§5.3 のキー・初期値と一致
- Tune 設定が `tuning.optuna.params` / `tuning.optuna.space` 形式で保存される
- `apply_best_params` / `request_inference_plot` を含む BLUEPRINT §3.6 action 一覧が実装済み
- Python API が BLUEPRINT §4.1 の戻り値契約（チェーン・型）を満たす
- 追加テストが CI で通過し、再発を検知できる

---

### Phase 10: 2026-03-10 要件監査の残開発（実画面確認ベース） ⚠️

**目標:** Notebook 実画面監査で確認した BLUEPRINT 乖離を解消し、モンキーパッチ不要で `idle → data_loaded → running → completed/failed` が再現できる状態にする。

#### 10-1. 既定 Fit 導線の恒常修正（`config_version` 欠落解消） ✅

- ✅ `widget.py:73` — `config.setdefault("config_version", 1)` でフォールバック実装済み
- ✅ `ConfigTab.tsx:239-243` — `config_version` を読み取り専用表示
- ✅ Fit 実行前バリデーション実装済み（`widget.py:390-398`）

#### 10-2. Adapter の LizyML 実ランタイム互換性修正（`available_plots`） ✅

- ✅ `adapter.py:183-188` — `model._cfg.task` を優先し `_widget_config` フォールバック付き
- ✅ `contextlib.suppress` で calibration / tuning 判定を防御的に実装

#### 10-3. Tune Search Space 契約整合（UI値→LizyML型） ✅

- ✅ `SearchSpace.tsx:55` — `NON_TUNABLE` セットで `name` を除外、object 型もフィルタ
- ✅ `ConfigTab.tsx:127-130` — Range/Choice が 1 つ以上ないと Tune ボタン disabled

#### 10-4. Results 表示の BLUEPRINT §5.4 完全整合 ✅

- ✅ `widget.py:521-548` — `_normalize_metrics` で evaluate_table を IS/OOS/OOS Std に変換
- ✅ `widget.py:430-434` — `fit_summary` に `fold_details` を含める
- ✅ `ScoreTable.tsx:13-15` — OOS Std は CV 時のみ列表示
- ✅ `ResultsTab.tsx:115-156` — Tune 完了: Optimization History → Best Params → Score 順
- ✅ `ResultsTab.tsx:134` — Apply to Fit ボタン → `apply_best_params` action

#### 10-5. Inference UI/Action 完全化（Return SHAP + 分布/SHAP） ⚠️

- ✅ `ResultsTab.tsx:287-294` — SHAP チェックボックス実装済み
- ✅ `ResultsTab.tsx:296-304` — Run Inference ボタン（ready 時のみ有効）
- ✅ `ResultsTab.tsx:316-342` — Prediction Distribution / SHAP Summary / Warnings の Accordion 構造あり
- ✅ `widget.py:283-301` — `run_inference` handler で `return_shap` 対応済み
- ✅ `widget.py:350-352` — `request_inference_plot` handler 実装済み
- ❌ adapter.py に `prediction-distribution` / `shap-summary` plot type が未定義 → Phase 11 へ

#### 10-6. 仕様ドキュメント同期（Action/traitlet 契約差分） ⚠️

- ✅ `set_task` action は widget.py に実装済み（BLUEPRINT §3.6 には既に記載あり）
- ❌ BLUEPRINT §3.6 の action テーブルに `import_yaml` / `export_yaml` / `raw_config` / `apply_best_params` が未記載 → Phase 11 へ
- 受け入れ条件
  - BLUEPRINT と実装の Action/traitlet 契約が 1対1 で対応し、監査時に解釈差分が出ない

#### 10-7a. チュートリアル Python API 実装（P-003 対応） ✅

- ✅ `src/lizyml_widget/widget.py` に以下を追加（P-003 承認済み 2026-03-11）:
  - `set_target(col)` — `_service.set_target()` を呼び `df_info` / `status` を更新
  - `fit(*, timeout=None)` — `threading.Event` で Fit 完了を待つブロッキングメソッド
  - `tune(*, timeout=None)` — 同様のブロッキングメソッド
  - `task` / `cv_method` / `cv_n_splits` / `df_shape` / `df_columns` — 読み取り専用プロパティ
- ✅ BLUEPRINT.md §4.1 に上記 API を追記
- ✅ HISTORY.md に P-003 Proposal を記録

#### 10-7. E2E 回帰テスト（Notebook 実行フロー） 🔲

- ❌ `tests/` に Notebook 統合テスト未追加
  - Data: target/task/CV 変更 → traitlet 反映
  - Fit: `idle → data_loaded → running → completed`
  - Tune: SearchSpace 最小構成で completed
  - Inference: ready → completed（SHAP off/on）
- 受け入れ条件
  - 実バックエンドを使った E2E で Phase 10 の主要不具合を検知できる

---

### Phase 11: 仕様同期 + 残 plot 実装（2026-03-11 監査結果） 🔲

**目標:** 2026-03-11 の要件監査で発見した乖離を解消する。

#### 11-1. BLUEPRINT §3.6 Action テーブル更新

- BLUEPRINT.md §3.6 に以下の action を追記:
  - `import_yaml` — `{"content": "yaml_string"}` — YAML 文字列を読み込み Config に反映
  - `export_yaml` — `{}` — 現在の Config を YAML でダウンロード（msg:custom で返す）
  - `raw_config` — `{}` — フル Config の YAML テキストを取得（msg:custom で返す）
  - `apply_best_params` — `{"params": {...}}` — Tune の Best Params を Fit の model.params にマージ

#### 11-2. Inference plot type 実装

- `src/lizyml_widget/adapter.py`
  - `plot_methods` に `prediction-distribution` を追加（推論結果の分布ヒストグラム）
  - `plot_methods` に `shap-summary` を追加（SHAP 値のサマリープロット）
  - Inference 用プロットは fit モデルではなく inference 結果に基づくため、別の生成パスが必要な可能性あり

#### 11-3. esbuild JSX フラグの BLUEPRINT 記載更新

- BLUEPRINT.md §8.1 の `--jsx-factory=h --jsx-fragment=Fragment` を `--jsx=automatic --jsx-import-source=preact` に更新（実装に合わせる）

#### 11-4. index.tsx の lzw-root クラス付与

- 現在 `el.classList.add("lzw-root")` を呼んでいない（BLUEPRINT §7.3 との微差）
- App.tsx 内の `<div class="lzw-root">` で代替されているが、BLUEPRINT に合わせるなら index.tsx に追加

#### 11-5. adapter.load_model() / model_info() の実装（低優先）

- 現在 `NotImplementedError`。Widget UI からは未使用だが Protocol 完全実装のため将来対応

#### 11-6. LightGBM `model.params` 初期値 pre-populate（BLUEPRINT §5.3 仕様変更）

- `src/lizyml_widget/service.py`
  - `build_config()` の `model.params` 初期値として LightGBM task 非依存デフォルトを設定
    - `n_estimators=1500`, `learning_rate=0.001`, `max_depth=5`, `max_bin=511`,
      `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=10`,
      `lambda_l1=0.0`, `lambda_l2=0.000001`, `first_metric_only=false`, `verbose=-1`
  - `set_target()` / `set_task()` 呼び出し時に task 依存項目（`objective`・`metric`）を `model.params` に追加・上書き
    - `objective`: regression=`"huber"`, binary=`"binary"`, multiclass=`"multiclass"`
    - `metric`: regression=`["huber","mae","mape"]`, binary=`["auc","binary_logloss"]`, multiclass=`["auc_mu","multi_logloss"]`
  - ユーザーが params を変更済みの場合はその値を優先（上書きしない）
  - `auto_num_leaves` 変更時の排他制御:
    - `true` → `params.num_leaves` を params から除去、`num_leaves_ratio` を表示
    - `false` → `params.num_leaves=256` を params に追加（未設定の場合のみ）、`num_leaves_ratio` を非表示
  - `verbose` は params KVEditor から外し、`model.params.verbose` として独立フィールドで管理
- `js/src/tabs/ConfigTab.tsx` または `js/src/components/DynForm.tsx`
  - `auto_num_leaves` の値に応じて `Num Leaves Ratio` ↔ `Num Leaves` を条件切替表示
  - `verbose` を Model セクション下部に独立 NumberInput として表示
- `js/src/components/SearchSpace.tsx`
  - Search Space テーブルを §5.3 の全パラメータ（LGBMParam + LGBMConfig）で pre-populate する
  - 初期モードはすべて Fixed（Fit 初期値をそのまま保持）
  - Range / Choice に変更したパラメータのみ `tuning.optuna.space` に書き込む（Fixed は空 dict 相当）
  - task 依存パラメータ（objective・metric）は task 確定後に更新
- `js/src/components/DynForm.tsx` / `widget.css`
  - KVEditor の既存実装で描画できることを確認（追加実装不要の見込み）

#### 11-7. Fit 初期導線の `model.name` 欠落修正（2026-03-12 監査）

- `src/lizyml_widget/widget.py`
  - `load()` 時の初期 config 生成で `model.name="lgbm"` を必ず保持する
  - `_extract_defaults()` を `oneOf` / `const`（discriminator schema）に対応させる、または後段で明示補完する
- `src/lizyml_widget/service.py`
  - `build_config()` で `model` が存在する場合でも `name` 欠落時は補完する（`model.params` のみ状態を許容しない）
- 受け入れ条件
  - `w.load(df, target="y")` 直後の `w.get_config()["model"]["name"] == "lgbm"`
  - ユーザー未編集で `fit` を実行して `status="completed"` まで到達する

#### 11-8. Model Type 表示と実データの整合性確保（2026-03-12 監査）

- `js/src/tabs/ConfigTab.tsx`
  - `Model Type` 表示を「実 config 値と一致」させる（UI フォールバック表示だけで欠落を隠さない）
  - `model.name` が欠落している異常状態を UI で判別可能にする（表示警告 or disabled）
- 受け入れ条件
  - `model.name` 欠落時に UI が正常値を偽装しない
  - `model.name` 設定時は `"lgbm"` と一致表示される

#### 11-9. バリデーション可観測性と回帰テスト強化（2026-03-12 監査）

- `src/lizyml_widget/adapter.py`
  - `validate_config()` の返却に、可能な範囲で validation detail（loc/type/msg）を保持する
- `tests/test_widget_api.py` / `tests/test_service.py`
  - 実 schema（`oneOf + discriminator`）を使った回帰テストを追加し、`model.name` 欠落を検知する
  - 「デフォルト config のまま fit 可能」を E2E 系テストに追加
- 受け入れ条件
  - `VALIDATION_ERROR` 発生時に根因キーが特定できる
  - 既知不具合（`model.name` 欠落）が自動テストで再検知可能

**完了条件:**
- BLUEPRINT §3.6 の action テーブルと実装が 1対1 で対応する
- Inference 完了後に Prediction Distribution / SHAP Summary の Plotly プロットが表示される
- BLUEPRINT §7.3 / §8.1 の記載と実装が一致する
- `w.load(df, target="y")` 直後の Model タブ Model セクションに LightGBM デフォルト params（n_estimators, learning_rate 等）が表示される
- task 変更時に objective / metric が params に自動設定される

---

### Phase 12: UI ユーザビリティ改善（BLUEPRINT §5.3 追加要件） 🔲

**目標:** マウス操作性向上、Calibration バグ修正、SearchSpace Choice モード拡充。

#### 12-1. Widget 高さリサイズ対応

- `js/src/widget.css`: `.lzw-root` の `height: 620px` → `min-height: 620px; resize: vertical; overflow: auto` に変更
- ドラッグで高さを拡大できる（620px を下限として拡大のみ許可）

#### 12-2. Fit / Tune サブタブバーのスティッキー化

- `js/src/tabs/ConfigTab.tsx`: サブタブバー（`[Fit] [Tune]` ボタン + 実行ボタン）を `position: sticky; top: 0; z-index: 1` にする
- スクロールしても Fit / Tune ボタンと実行ボタンが常に表示される

#### 12-3. SearchSpace Objective / Metric の Choice モード実装

- `js/src/components/SearchSpace.tsx`:
  - `modesForParamType("string")` を `["fixed", "choice"]` に変更（objective / metric のみ）
  - Choice モード UI: OBJECTIVE_OPTIONS[task] / METRIC_OPTIONS[task] のチェックボックスリストを表示
  - `config.choices` に選択した `string[]` を格納

#### 12-4. auto_num_leaves が Choice の場合に num_leaves 行を表示

- `js/src/components/SearchSpace.tsx`:
  - `auto_num_leaves` が `choice` モード時にも `num_leaves` 行（Fixed / Range / Choice）を追加表示

#### 12-5. Tune Settings — Optuna 最適化 Metric 指定

- `js/src/tabs/ConfigTab.tsx` Tune Settings 内:
  - `tuning.optuna.params.metric` フィールドを追加（LizyML スキーマ確認後）
  - task 別 METRIC_OPTIONS + `null`（デフォルト）のセレクトを表示

#### 12-6. 数値入力フィールドの step 設定

- `js/src/tabs/ConfigTab.tsx` TypedParamsEditor: 各数値フィールドに BLUEPRINT §5.3 フォーム動的生成 の step 値を設定
- `js/src/components/SearchSpace.tsx`: Range モードの low / high inputs に step 値を設定

#### 12-7. Evaluation Metrics を選択式に

- `js/src/tabs/ConfigTab.tsx`:
  - `evaluation.metrics` を DynForm の array タグ入力から task 依存チェックボックスグループに変更
  - METRIC_OPTIONS[task] を選択肢として使用

#### 12-8. Inner Validation を選択式に

- `js/src/tabs/ConfigTab.tsx`:
  - `training.early_stopping.inner_valid` を text input からセレクトに変更
  - 選択肢: `null`（auto）+ split 構成から導出した validation 名

#### 12-9. Calibration トグルバグ修正

- `js/src/tabs/ConfigTab.tsx`:
  - Calibration トグル（Accordion ヘッダー内）のクリックが Accordion 開閉と競合する問題を修正
  - トグルの `onChange` ハンドラに `e.stopPropagation()` を適用（または Accordion ヘッダー構造を分離）

#### 12-10. Tune Settings — Direction 自動化（削除）

- `js/src/tabs/ConfigTab.tsx`:
  - Tune Settings から `direction` の UI 項目を削除
  - direction は Optuna metric に応じてバックエンド（LizyML / Adapter）側で自動決定される
  - `tuning.optuna.params.direction` キーを config から除外

#### 12-11. Tune Settings — Timeout 削除

- `js/src/tabs/ConfigTab.tsx`:
  - Tune Settings から `timeout` の UI 項目を削除
  - `tuning.optuna.params.timeout` キーを config から除外

**完了条件:**
- Widget 下端ドラッグで高さが拡大できる
- Fit / Tune サブタブバーと実行ボタンがスクロール後も表示される
- SearchSpace の objective / metric を Choice モードに変更できる
- Tune Settings に Optuna 最適化 metric のセレクトがある
- 全数値入力フィールドに正しい step が設定されている
- Evaluation metrics がチェックボックスグループで表示される
- Calibration トグルが正常に ON / OFF できる
- Tune Settings に direction / timeout が表示されない
