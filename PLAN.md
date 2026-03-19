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
| 6 | Inference | load_inference() → 推論実行 → 結果表示 | ✅ 完了 |
| 7 | Config Import/Export + Python API | YAML I/O + プログラム API の完成 | ⚠️ ほぼ完了（load_model/model_info 未実装） |
| 8 | 品質・配布 | CI・テスト拡充・PyPI 配布 | ✅ 完了 |
| 9 | BLUEPRINT 乖離解消 | 要件監査是正 | ✅ 完了 |
| 10 | 実画面確認ベース残開発 | Notebook 実画面監査の乖離解消 | ✅ 完了 |
| 11 | 仕様同期 + 残 plot 実装 | 監査で発見した乖離の解消 | ✅ 完了 |
| 12 | UI ユーザビリティ改善 | マウス操作性向上・Calibration バグ修正・SearchSpace Choice 拡充 | ✅ 完了 |
| 13 | 2026-03-12 要件監査 乖離解消 | BLUEPRINT/実装の乖離是正 + エラーコード整備 | ✅ 完了 |
| 14 | Tune 実行失敗の根治 | Tune を Python API / Widget UI で安定完了させる | ✅ 完了 |
| 15 | Widget ウィンドウ調整の実効化 | Notebook 上の高さ調整を実際の表示領域拡張に反映させる | ✅ 完了 |
| 16 | Tune Mode UX 改善 | Search Space の Mode 選択をセグメントボタン化して操作コストを下げる | ✅ 完了 |
| 17 | Numeric Input UX 改善 | 数値入力を大型 `- / +` ステッパーへ統一し操作性を向上する | ✅ 完了 |
| 18 | Table Grid UX 改善 | Column Settings / Search Space を CSS Grid + minmax() に置換し列幅を自動調整する | ✅ 完了 |
| 19 | Apply to Fit 同期強化 | Tune 実行時設定を Fit 画面へフル同期し再現性を担保する | ✅ 完了 |
| 20 | Tune Metric UX 改善 | Tune Settings の metric セグメントから Default を削除し選択を明確化する | ✅ 完了 |
| 21 | Data 選択 UX 改善 | Task / CV Strategy をチップ選択へ統一して操作負荷を下げる | ✅ 完了 |
| 22 | Config 契約確定 + 検証強化 | UI→LizyML の Config 受け渡し仕様を確定し、仕様逸脱を自動検知できるテストを整備する | ✅ 完了 |
| 23 | 入力コントロール統一の追補 | 数値幅固定・セグメント/チップ化・Inner Valid表示を統一して操作一貫性を高める | ✅ 完了 |
| 24 | Widget / Service 疎結合化 | config 初期化・実行準備を Service に集約し、Widget から private 境界越えを除去する | ✅ 完了 |
| 25 | Backend Contract 駆動の完全疎結合化 | backend 固有 UI/Config 知識を Adapter contract へ集約し、UI を generic renderer 化する | 📝 設計完了（未実装） |
| 26 | Canonical Config 経路統一 + `inner_valid` 契約整合化 | config canonicalization の単一路線化、Validation 診断改善、Phase 25 残課題の追補 | ⚠️ 部分実装（追補要） |
| 27 | Google Colab 互換ポーリング | BG スレッド traitlet 同期の Colab 制約を JS ポーリングで回避 | 📝 計画済み |
| 28 | ダークモード対応 | CSS 変数化 + `prefers-color-scheme` 対応 + Plotly テーマ追従 | 📝 計画済み |

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
    - Task: 自動判定結果の表示（当時はドロップダウン。Phase 21 でチップ選択へ更新）
  - Column Settings テーブル（`js/src/components/ColumnTable.tsx`）
    - Column / Uniq / Excl チェックボックス / Type ドロップダウン
    - バッジ表示（[ID], [Const]）
    - 変更時に `update_column` action 送信
  - Cross Validation セクション
    - Strategy 選択 UI / Folds 数値入力 / Group column（条件付き表示）（当時はドロップダウン。Phase 21 でチップ選択へ更新）
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

### Phase 6: Inference ✅

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

### Phase 8: 品質・配布 ✅

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
Phase 5 ✅ Phase 6 ✅  Phase 7 ⚠️
(Tune)     (Inference) (Import/Export + API)
  │          │          │
  └──────────┴──────────┘
             │
  ┌──────────┼──────────┐
  ▼          ▼          ▼
Phase 9 ✅ Phase 10 ✅  Phase 11 ✅
(乖離解消) (実画面修正) (仕様同期+残plot)
             │          │
             └──────────┘
                  │
        ┌────────┼────────┐
        ▼        ▼        ▼
  Phase 12 ✅  Phase 13 ✅
  (UI改善)   (監査乖離解消)
        │        │
        └────────┘
             │
             ▼
        Phase 8 ✅ (品質・配布)
```

Phase 11 は Phase 10 の残作業 + 監査乖離の解消。Phase 12 と Phase 13 は独立して並行実施可能。Phase 8 は全フェーズ完了後に実施。

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

### Phase 10: 2026-03-10 要件監査の残開発（実画面確認ベース） ✅

**目標:** Notebook 実画面監査で確認した BLUEPRINT 乖離を解消し、モンキーパッチ不要で `idle → data_loaded → running → completed/failed` が再現できる状態にする。

#### 10-1. 既定 Fit 導線の恒常修正（`config_version` 欠落解消） ✅

- ✅ `widget.py:73` — `config.setdefault("config_version", 1)` でフォールバック実装済み
- ✅ `ConfigTab.tsx:239-243` — `config_version` を読み取り専用表示
- ✅ Fit 実行前バリデーション実装済み（`widget.py:390-398`）

#### 10-2. Adapter の LizyML 実ランタイム互換性修正（`available_plots`） ✅

- ✅ `adapter.py:183-188` — `model._cfg.task` を優先し `_widget_config` フォールバック付き
- ✅ `contextlib.suppress` で calibration / tuning 判定を防御的に実装

#### 10-3. Tune Search Space 契約整合（UI値→LizyML型） ⚠️ Phase 14 で再対応

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

### Phase 11: 仕様同期 + 残 plot 実装（2026-03-11 監査結果） ✅

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

### Phase 12: UI ユーザビリティ改善（BLUEPRINT §5.3 追加要件） ✅

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
  - task 別 METRIC_OPTIONS + `null`（デフォルト）の選択 UI を表示（当時はセレクト実装。Phase 20 でセグメントへ更新）

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
- Tune Settings に Optuna 最適化 metric の選択 UI がある（Phase 20 でセグメント化予定）
- 全数値入力フィールドに正しい step が設定されている
- Evaluation metrics がチェックボックスグループで表示される
- Calibration トグルが正常に ON / OFF できる
- Tune Settings に direction / timeout が表示されない

---

### Phase 13: 2026-03-12 要件監査 乖離解消 ✅

**目標:** 2026-03-12 の全体要件監査で発見された BLUEPRINT と実装の乖離を解消し、仕様ドキュメントと実装を完全に一致させる。

Phase 11 で対応済みの項目（BLUEPRINT §3.6 Action テーブル更新・Inference plot type 実装）は除く。Phase 11 未対応かつ Phase 12 と独立の乖離を本フェーズで対応する。

#### 13-1. Apply to Fit ボタンのタブ切替実装

- **乖離:** BLUEPRINT §5.4 は「Fit サブタブへコピーして切り替え」と記載するが、現在の実装はパラメータ適用のみでタブ切替しない
- **対応:**
  - `js/src/tabs/ResultsTab.tsx`: Apply to Fit ボタン押下時にコールバックで親へ通知
  - `js/src/App.tsx`: コールバック受信で `activeTab → "Model"` に切替え
  - `js/src/tabs/ConfigTab.tsx`: 外部から `subTab` を制御可能にする（`initialSubTab` prop、または effect で Fit に戻す）
- **受け入れ条件:** [Apply to Fit ▸] 押下 → Model タブ Fit サブタブに自動切替え、パラメータが反映済み

#### 13-2. Results タブ有効条件の BLUEPRINT 更新

- **乖離:** BLUEPRINT §5.1 は Results タブの有効条件を「`completed` または `failed` 以降」と記載するが、実装は `running` 状態でも有効（進捗表示のため）
- **対応:** BLUEPRINT §5.1 のタブ有効条件を実装に合わせて更新
  - 現行: `status` が `completed` または `failed` 以降（一度でも実行済み）
  - 変更後: `status` が `running` / `completed` / `failed` のいずれか、または過去に一度でも実行済み（`job_index > 0`）
- **受け入れ条件:** BLUEPRINT §5.1 と `App.tsx` のタブ有効条件ロジックが一致する

#### 13-3. Plots 表示条件の BLUEPRINT 更新

- **乖離 A:** BLUEPRINT §5.4 の Plots セレクタに `probability-histogram` は「binary」のみ記載だが、実装は `has_calibration` を追加条件としている
- **乖離 B:** BLUEPRINT §5.4 に `roc-curve` は「binary」のみ記載だが、実装は `multiclass` でも表示する
- **対応:** BLUEPRINT §5.4 の表示条件テーブルを実装に合わせて更新

  | plot_type | 更新後の表示条件 |
  |-----------|-----------------|
  | `roc-curve` | `binary` / `multiclass` |
  | `calibration` | `binary` + calibration 有効 |
  | `probability-histogram` | `binary` + calibration 有効 |

- **受け入れ条件:** BLUEPRINT §5.4 と `adapter.py:available_plots()` の条件が一致する

#### 13-4. エラーコード `NO_DATA` / `NO_TARGET` の明示化

- **乖離:** BLUEPRINT §6.1 に `NO_DATA` / `NO_TARGET` / `INTERNAL_ERROR` を定義するが、実装では Service 層の `ValueError` が `BACKEND_ERROR` として汎化される
- **対応:**
  - `src/lizyml_widget/widget.py`
    - `_run_job()` で `self._service._df is None` を事前チェックし `{"code": "NO_DATA", "message": "..."}` を返す
    - `_run_job()` で `self._service._df_info.get("target") is None` を事前チェックし `{"code": "NO_TARGET", "message": "..."}` を返す
    - catch-all の `Exception` に対して既存の `BACKEND_ERROR` に加え `INTERNAL_ERROR` を適切に使い分ける（Adapter 由来 → `BACKEND_ERROR`、それ以外 → `INTERNAL_ERROR`）
  - `tests/test_widget_api.py`
    - データ未ロードで fit → `NO_DATA` エラーのテスト
    - ターゲット未選択で fit → `NO_TARGET` エラーのテスト
- **受け入れ条件:** BLUEPRINT §6.1 の全エラーコード（`NO_DATA` / `NO_TARGET` / `VALIDATION_ERROR` / `BACKEND_ERROR` / `CANCELLED` / `INTERNAL_ERROR`）が実装内で使い分けられている

#### 13-5. ProgressView の fold_results 送信（低優先・バックエンド依存）

- **乖離:** BLUEPRINT §5.4 は Fold 別スコアのリアルタイム表示を想定するが、LizyML の `model.fit()` が Fold コールバックでスコアを返さないため `fold_results` が常に空
- **対応:** LizyML 側の Fold コールバック API を調査し、利用可能なら Adapter 層で `on_progress` コールバックのペイロードに `fold_results` を含める
  - 利用不可の場合: BLUEPRINT §5.4 の該当記述に「バックエンドの Fold コールバック対応に依存。LizyML 初期版では省略」と注記
- **優先度:** 低（バックエンド側の機能追加待ち）

**完了条件:**
- [Apply to Fit ▸] 押下で Model タブ Fit サブタブに自動切替えされる
- BLUEPRINT §5.1 のタブ有効条件が実装と一致
- BLUEPRINT §5.4 の plot 表示条件が `adapter.py` と一致
- `NO_DATA` / `NO_TARGET` エラーが専用コードで返却され、テストでカバーされている

---

### Phase 14: Tune 実行失敗の根治（2026-03-12 深掘り調査） ✅

**目標:** Tune を「Python API / Widget UI どちらの導線でも」安定して完了させ、`status="completed"` まで到達できる状態に戻す。

**深掘り調査で確定した再現パターン（2026-03-12）:**

- R1: `w.load(...).set_target(...); w.tune()`（`tuning` 未設定）
  - 失敗: `[CONFIG_INVALID] No tuning configuration found...`
- R2: UI の Search Space で Range/Choice を 1件設定して Tune 実行
  - 失敗: `[CONFIG_INVALID] Unknown search space type '' ...`
  - 原因: `tuning.optuna.space` が `mode=range/choice` 形式で保存されるが、LizyML は `type=float/int/categorical` を要求
- R3: 有効な `tuning` を与えて Tune 本体は成功
  - その後失敗: `[MODEL_NOT_FIT] Model has not been fitted. Call fit() first.`
  - 原因: Tune 後に `evaluate_table()` / `split_summary()` を呼び、Fit 済みを前提にしている
- R4: Tune-only 実行後の `available_plots`
  - `learning-curve` 等の Fit 依存プロットが列挙されるが、実際に取得すると `MODEL_NOT_FIT`

#### 14-0. 変更ゲート判定（実装前）

- `tune()` 実行時の設定補完方針（`tuning` 未設定時の扱い）は Python API のデータフロー変更に該当するため、先に HISTORY.md に Proposal を追加する
  - 候補: P-004「Tune 起動時の `tuning` デフォルト補完」
- Proposal 承認後に実装へ進む

#### 14-1. Tune 起動時の `tuning` デフォルト補完（R1 対応）

- `src/lizyml_widget/widget.py`
  - `_run_job("tune")` 実行時、`full_config["tuning"]` が `None` / 欠落なら最小有効構成を補完
    - 例: `{"optuna": {"params": {"n_trials": 50}, "space": {}}}`
  - 補完後の config を `validate_config()` に通す
- `notebooks/tutorial.ipynb`
  - Tune セルの前に、`tuning` 明示設定が不要であること（または必要な場合の最小設定）を注記
- 受け入れ条件
  - `w.load(df).set_target("y"); w.tune()` で `CONFIG_INVALID(No tuning configuration...)` が発生しない

#### 14-2. Search Space 契約変換の修正（R2 対応）

- `js/src/components/SearchSpace.tsx`
  - `mode` は UI 内部状態として保持しつつ、`onChange` で Python 側に渡す値は LizyML 契約へ変換する
    - Range（float）→ `{"type":"float","low":...,"high":...,"log":...}`
    - Range（integer）→ `{"type":"int","low":...,"high":...,"log":...}`
    - Choice → `{"type":"categorical","choices":[...]}`
    - Fixed → `space` から key を削除
  - Choice が空配列の場合は保存しない（または UI 側でエラー表示）
- `src/lizyml_widget/adapter.py`（防御）
  - `validate_config()` に `tuning.optuna.space` の追加検証を実装し、`type/low/high/choices` 欠落を `VALIDATION_ERROR` として早期返却
  - 旧形式（`mode`）を検出したら明示的エラーメッセージを返す
- 受け入れ条件
  - Tune 実行で `Unknown search space type ''` が発生しない
  - 不正 Search Space は backend 実行前に `VALIDATION_ERROR` で止まる

#### 14-3. Tune 後の Fit 前提処理を撤去（R3 対応）

- `src/lizyml_widget/widget.py`
  - `job_type == "tune"` 分岐での `evaluate_table()` / `split_summary()` 呼び出しをガードする
  - Fit 未実行（`MODEL_NOT_FIT`）時は `fit_summary` を空のまま維持し、Tune 成功を失敗にしない
- `src/lizyml_widget/adapter.py`
  - 必要に応じて `is_fitted(model)` 相当の判定ヘルパーを追加し、Widget 側で利用
- 受け入れ条件
  - Tune 本体成功時に `status="completed"` になる
  - Tune-only 実行で `MODEL_NOT_FIT` が `status="failed"` の原因にならない

#### 14-4. `available_plots` の Fit/Tune 状態整合（R4 対応）

- `src/lizyml_widget/adapter.py`
  - Fit 依存プロット（`learning-curve`, `oof-distribution`, `roc-curve`, `residuals`, `feature-importance`, calibration系）は `model` が Fit 済みのときのみ追加
  - Tune-only の場合は `optimization-history` のみ（+将来必要な Tune 専用プロット）を返す
- `js/src/tabs/ResultsTab.tsx`
  - 返却された `available_plots` のみを選択可能にする（既存挙動の確認と不足があれば補強）
- 受け入れ条件
  - Tune-only 実行後に Fit 依存プロットを要求してエラーにならない

#### 14-5. Tune Settings `metric` のスキーマ追従（互換性修正）

- `js/src/tabs/ConfigTab.tsx`
  - `tuning.optuna.params.metric` は schema に存在する場合のみ表示・保存
  - `metric` 非対応 backend（例: lizyml 0.1.2）では UI 項目を非表示にする
- BLUEPRINT.md
  - 既存記載「LizyML スキーマでサポートされている場合のみ表示」を満たす実装条件を明文化
- 受け入れ条件
  - `metric` 非対応スキーマで Tune Settings を開いても `VALIDATION_ERROR` が増えない

#### 14-6. 回帰テストの増強（実バックエンド中心）

- `tests/test_widget_api.py`
  - `tune()` 呼び出し時の `tuning` 補完テスト
  - Tune 本体成功後に `status="completed"` を維持するテスト（`fit_summary` 空許容）
- `tests/test_adapter.py`
  - `available_plots()` が Tune-only で Fit 依存プロットを返さないこと
  - Search Space 追加検証（不正形式を `VALIDATION_ERROR` として返す）
- `tests/test_e2e.py`（実 backend または準実 backend）
  - Tune 最小構成（`space={}`）で completed
  - UI 相当 Search Space（Range/Choice）で completed

#### 14-7. ドキュメント同期

- `PLAN.md`（本フェーズ）
  - 10-3 を「契約整合未完了（再オープン）」として注記し、Phase 14 を正本にする
- `HISTORY.md`
  - Proposal / Decision / Migration を追記（14-0 の結果）
- `BLUEPRINT.md`
  - Tune ボタン有効条件と `space={}` 利用方針の矛盾を解消する
    - 方針候補A: Range/Choice 0件でも Tune 実行可（backend default space）
    - 方針候補B: Range/Choice 1件必須（`space={}` default運用を仕様から除外）
  - Decision は HISTORY に合わせて反映する

**完了条件:**

- Python API: `w.tune()` がデフォルト構成で `completed` まで到達する
- UI: Search Space を 1件以上設定した Tune が `completed` になる
- Tune-only 実行で `MODEL_NOT_FIT` により失敗しない
- `available_plots` が実行状態（fit/tune）と矛盾しない
- 追加した回帰テストで R1〜R4 を継続検知できる

---

### Phase 15: Widget ウィンドウ調整不具合の解消（2026-03-12 調査）✅

**目標:** Notebook（Jupyter / VS Code / Colab）上で Widget 下端リサイズが実表示領域に反映され、Model タブの作業領域が実際に拡張される状態にする。

**調査で確定した原因（2026-03-12）:**

- C1: `js/src/index.tsx` と `js/src/App.tsx` の双方で `.lzw-root` が適用され、`lzw-root` コンテナが二重化している
  - `index.tsx`: `el.classList.add("lzw-root")`
  - `App.tsx`: `<div class="lzw-root">`
  - 結果: 外側リサイズと内側表示領域が分離し、下部に空白が残る
- C2: `js/src/widget.css` の `.lzw-root` が依然として `height: 620px; overflow: hidden;` で固定されており、Phase 12-1 の計画値（`min-height + resize + overflow:auto`）と不一致
- C3: Phase 11-4（`index.tsx` への `lzw-root` 付与）と Phase 12-1（高さリサイズ方針）の整合が未完了のまま完了扱いとなり、回帰検知が漏れている

#### 15-1. レイアウト責務の一本化（root クラス二重適用の解消）

- `js/src/index.tsx` と `js/src/App.tsx` のどちらか一方のみを `.lzw-root` の所有者に統一する
- 推奨方針: anywidget host 要素（`el`）を `.lzw-root` とし、`App.tsx` 側は `lzw-app` など別クラスへ分離
- 受け入れ条件
  - Notebook 上で `.lzw-root` が 1 コンテナのみ
  - 二重 border / 二重スクロール領域が発生しない

#### 15-2. 高さリサイズ CSS 契約の再実装

- `js/src/widget.css` の `.lzw-root` を BLUEPRINT §5.1 と Phase 12-1 の意図に一致させる
  - 固定 `height: 620px` を廃止
  - `min-height: 620px` を下限として保持
  - `resize: vertical` + `overflow: auto` で下端ドラッグを有効化
- `.lzw-content` を含む内部 flex レイアウトに `min-height: 0` を適用し、親高さ変更時にスクロール領域が追従するようにする
- 受け入れ条件
  - 高さ拡張後、Model タブの見える行数が増える（空白だけ増えない）
  - Fit/Tune サブタブ sticky が高さ変更後も維持される

#### 15-3. Notebook 実行環境別の確認と最小再現手順の固定化

- `notebooks/tutorial.ipynb` に「リサイズ確認セル」を追加し、手動確認手順を明文化する
  - 手順例: Model タブで下端をドラッグ → 設定フォーム可視領域が増えることを確認
- 確認対象を明示する
  - JupyterLab（Linux/macOS）
  - VS Code Notebooks（Windows を優先）
- 受け入れ条件
  - 上記 2 環境で同一手順により再現しない（調整が実効する）

#### 15-4. 回帰防止テストの追加（静的契約 + UI 振る舞い）

- `tests/` にフロントエンド契約の静的テストを追加する
  - `App.tsx` と `index.tsx` の `.lzw-root` 重複を検知
  - `widget.css` に固定 `height: 620px` が再導入された場合に失敗
- 可能なら notebook 実行の E2E チェック（Phase 10-7）に「リサイズ後の表示領域拡張」を追加する
- 受け入れ条件
  - CI でレイアウト契約の退行を自動検知できる

#### 15-5. ドキュメント整合（完了定義の是正）

- `PLAN.md`
  - Phase 12-1 を「完了済み」前提で残さず、Phase 15 で再オープンした理由を明記（本節）
- `BLUEPRINT.md`
  - §7.3（anywidget エントリ）と §7.5（CSS スコープ）の例を、root クラス単一運用ルールに合わせる
- 受け入れ条件
  - BLUEPRINT / PLAN / 実装の `lzw-root` 責務が一致する

**完了条件:**

- Notebook 上で Widget 下端ドラッグにより、実作業領域（フォーム可視領域）が増加する
- `.lzw-root` の二重適用が解消され、二重スクロール/空白領域問題が再現しない
- Windows（VS Code Notebooks）で同現象が再発しないことを確認できる
- 回帰防止テストが追加され、固定高さと root 二重化の再発を CI で検知できる

---

### Phase 16: Tune Mode セグメントボタン化（2026-03-12 仕様更新）✅

**目標:** Search Space の Mode（Fixed / Range / Choice）をプルダウンからセグメントボタンに置き換え、Tune 設定の操作速度と視認性を改善する。

#### 16-1. SearchSpaceRow の Mode UI をセグメント化

- `js/src/components/SearchSpace.tsx`
  - 現在の Mode `<select>` を廃止し、行内セグメントボタン（2 or 3 分割）に置換
  - パラメータ型ごとの許可モードのみ表示（例: 数値は Fixed/Range、boolean は Fixed/Choice）
  - クリック 1 回でモード切替（Fixed → Range / Choice）
- 受け入れ条件
  - Mode 切替にドロップダウン操作が不要
  - 現在モードが視覚的に常時判別できる

#### 16-2. スタイル・レイアウト調整

- `js/src/widget.css`
  - Search Space 表の Mode 列向けに `.lzw-segment` 系スタイルを追加
  - 既存テーブル幅で折り返しや崩れが出ないよう調整（Notebook 横幅で確認）
  - active / hover / disabled 状態を定義
- 受け入れ条件
  - Data/Model/Results の既存スタイルと競合しない
  - 狭いセル幅でも Mode UI が崩れない

#### 16-3. アクセシビリティとキーボード操作

- セグメント各ボタンに `type="button"` と適切な `aria-pressed` を付与
- フォーカスリングを視認可能にし、Tab 操作で全モードに到達可能にする
- 受け入れ条件
  - マウスなしでも Mode 切替できる
  - active 状態がスクリーンリーダー属性で判別できる

#### 16-4. 回帰テストと仕様同期

- `tests/` に Search Space Mode UI の回帰テストを追加
  - Mode 切替で `tuning.optuna.space` 変換結果（`type=float/int/categorical`）が維持されること
  - Fixed へ戻したとき key が `space` から除去されること
- `BLUEPRINT.md` の §5.3（Tune サブタブ）と実装を一致させる
- 受け入れ条件
  - 既存 Tune フロー（実行成功・検証）を壊さない
  - CI で Mode UI 変更の退行を検知できる

**完了条件:**

- Search Space の Mode 列がセグメントボタンになっている
- 1 クリックで Mode が切り替わり、Config 入力欄が即時追従する
- Tune 実行前後で `tuning.optuna.space` 契約（`type` ベース）が維持される
- キーボード操作と回帰テストで再発防止できる

---

### Phase 17: 数値入力の大型ステッパー化（2026-03-12 仕様更新）✅

**目標:** `number` / `integer` フィールドのブラウザ標準スピナー依存をやめ、押しやすい大型 `- / +` ステッパー（直接入力併用）へ統一する。

#### 17-1. 共通 NumericStepper コンポーネント導入

- `js/src/components/` に共通の数値入力コンポーネント（仮: `NumericStepper`）を追加
  - 左右に大型 `-` / `+` ボタンを配置
  - 中央に直接入力可能な数値フィールドを維持
  - `step`, `min`, `max` を既存仕様に合わせて反映
- 受け入れ条件
  - 小さい上下スピナー操作なしで数値を増減できる
  - 直接入力とボタン操作の両方が使える

#### 17-2. Fit / Tune / Data の数値入力を置換

- `js/src/tabs/ConfigTab.tsx`
  - TypedParamsEditor, Training, Tune Settings, Log Output の数値項目を `NumericStepper` に置換
- `js/src/components/SearchSpace.tsx`
  - Fixed モード数値入力と Range の `low` / `high` を `NumericStepper` 化（Tune Search Space を含む）
- `js/src/tabs/DataTab.tsx` / `js/src/components/ColumnTable.tsx`（該当箇所）
  - CV Folds / Gap / Purge Gap / Embargo / random_state など数値項目を置換
- 受け入れ条件
  - 主要数値項目がすべて大型 `- / +` ステッパーで操作できる
  - 既存の `step` 挙動（Phase 12-6）を維持する

#### 17-3. スタイルとアクセシビリティ

- `js/src/widget.css`
  - `.lzw-stepper` 系スタイルを追加（ヒットエリア拡大、active/disabled/focus）
  - 数値入力欄の `min-width` を拡張（例: `8ch`）し、桁数の多い値でも欠けないようにする
  - ブラウザ標準 number spinner を非表示化
- キーボード操作（Tab / ArrowUp / ArrowDown）とスクリーンリーダー属性を確認
- 受け入れ条件
  - マウス操作で押しにくさが解消される
  - 数値欄の桁欠けが発生しない
  - キーボード操作性と視認性を損なわない

#### 17-4. 回帰テストとドキュメント同期

- `tests/` に数値入力コンポーネントの回帰テストを追加
  - `+/-` クリックで `step` どおりに増減
  - `min/max` 境界を超えない
  - 直接入力時の値検証が従来と同等
- `BLUEPRINT.md` §5.2 / §5.3 の数値入力仕様と一致を確認
- 受け入れ条件
  - Tune / Fit / Data の既存機能回帰がない
  - CI で数値入力 UI の退行を検知できる

**完了条件:**

- 数値入力 UI がブラウザ標準スピナーから大型 `- / +` ステッパーへ置換されている
- Data / Fit / Tune / Search Space の数値項目で一貫した操作体験を提供できる
- `step` / `min` / `max` 契約を維持したまま操作性が向上している
- 数値入力欄の表示幅が十分で、値が見切れない
- 回帰テストで増減ロジックと境界条件を継続検知できる

---

### Phase 18: CSS Grid + `minmax()` で列幅自動調整 ✅

**目標:** Column Settings / Search Space を `<table>` 依存から CSS Grid へ置換し、`minmax()` による列幅の完全自動調整を実現する。

#### 18-1. Column Settings の Grid 置換

- `js/src/components/ColumnTable.tsx`
  - `<table>` / `<tr>` / `<td>` 構造を Grid 行コンポーネントへ置換
  - ヘッダー行・データ行で同一 `grid-template-columns` を共有する
- `js/src/widget.css`
  - `.lzw-columns-grid` に `grid-template-columns: minmax(14rem, 2.4fr) minmax(4.5rem, 0.8fr) minmax(4.5rem, 0.8fr) minmax(12rem, 1.8fr)` を適用
  - セル内コントロールは `width: 100%` を基本とする
- 受け入れ条件
  - Column / Uniq / Excl / Type の列幅が内容に応じて自然に伸縮する
  - 既存アクション（`update_column`）と列意味が不変

#### 18-2. Search Space の Grid 置換

- `js/src/components/SearchSpace.tsx`
  - Search Space の行表示を Grid 化（Param / Mode / Config）
  - Mode 切替と `tuning.optuna.space` 変換ロジックは既存のまま維持
- `js/src/widget.css`
  - `.lzw-search-space-grid` に `grid-template-columns: minmax(14rem, 2.6fr) minmax(9rem, 1.2fr) minmax(16rem, 2.2fr)` を適用
  - Config 列コントロールを `width: 100%` で収める
- 受け入れ条件
  - Search Space の列が過不足なく自動調整される
  - Range / Choice / Fixed 切替の挙動が回帰しない

#### 18-3. 狭幅フォールバックと確認

- Notebook（JupyterLab / VS Code Notebooks）で横幅の異なるセルを確認
- `.lzw-table-wrap` / `.lzw-search-space` の `overflow-x: auto` を維持し、狭幅時は横スクロールへ退避
- 必要に応じて CSS 契約テスト（Grid テンプレート破壊の検知）を追加
- 受け入れ条件
  - 広い画面で間延びが解消される
  - 狭い画面でレイアウト崩れなく操作できる

**完了条件:**

- Column Settings / Search Space が CSS Grid + `minmax()` で安定動作する
- 列幅調整が手動チューニング不要で、内容ベースに自動追従する
- Notebook 環境差分でも可読性と操作性が維持される

---

### Phase 19: Apply to Fit の Tune 設定フル同期 ✅

**目標:** Tune 完了後の [Apply to Fit ▸] 実行時に、Fit 画面の全パラメータを Tune 実行時設定と一致させる。

#### 19-1. Tune 実行時 config スナップショット保持

- `src/lizyml_widget/widget.py` / `src/lizyml_widget/service.py`
  - Tune 実行開始時の有効 config（build 後）を内部状態として保持
  - 後続の Apply to Fit で参照できる形にする
- 受け入れ条件
  - Tune 完了後に「どの設定で Tune したか」を復元可能

#### 19-2. `apply_best_params` の適用順序を更新

- `apply_best_params` action の処理を以下に統一:
  1. Tune 実行時 config スナップショットを復元
  2. `best_params` を `model.params` に上書き
  3. Model タブ Fit サブタブへ切り替え
- 旧仕様（`model.params` への単純マージ）を本仕様で置換
- 受け入れ条件
  - Apply to Fit 後、Fit 画面の全項目（model/training/evaluation/calibration/output_dir 等）が Tune 実行時設定と一致する

#### 19-3. UI 同期と回帰テスト

- `js/src/tabs/ResultsTab.tsx` / `js/src/App.tsx` / `js/src/tabs/ConfigTab.tsx`
  - Apply to Fit 実行後、Fit サブタブに遷移し、復元済み config を即時表示する
- `tests/test_widget_api.py` / `tests/test_e2e.py`
  - 「Tune 実行 → Apply to Fit」で Fit 画面設定が Tune 実行時と一致する回帰テストを追加
  - `best_params` が `model.params` に反映されることを検証
- 受け入れ条件
  - 既存の Tune 完了導線を壊さず、Apply to Fit の再現性が保証される

**完了条件:**

- [Apply to Fit ▸] 実行後、Fit 画面が Tune 実行時設定と一致する
- `best_params` が `model.params` に適用された状態で再学習を開始できる
- テストで同期退行（部分マージへの後退）を検知できる

---

### Phase 20: Tune Settings `metric` のセグメントボタン化（2026-03-13 仕様更新）✅

**目標:** Tune Settings の `metric` 選択 UI をセグメントボタン化し、`Default` 選択肢を廃止して選択意図を明確化する。

#### 20-1. Tune Settings の `metric` 入力 UI 置換

- `js/src/tabs/ConfigTab.tsx`
  - `tuning.optuna.params.metric` の `<select>` を廃止
  - `METRIC_OPTIONS[task]` のみを単一選択セグメントボタンに置換（`Default` なし）
  - 初期選択は `METRIC_OPTIONS[task][0]` を採用し、選択値を `tuning.optuna.params.metric` に保存
- 受け入れ条件
  - 1 クリックで metric が切り替わる
  - 選択中の metric が常時強調表示される

#### 20-2. スキーマ互換条件の維持

- `js/src/tabs/ConfigTab.tsx`
  - 既存仕様どおり、backend schema が `tuning.optuna.params.metric` をサポートする場合のみ表示
  - 非サポート時は UI を非表示とし、保存 payload に `metric` を含めない
- `js/src/components/SearchSpace.tsx` / `js/src/tabs/ConfigTab.tsx`
  - `tuning.optuna.space.metric` は `model.params.metric` 探索用の Search Space エントリとして扱う（Tune Settings の `params.metric` とは別概念）
  - Search Space で `metric` を Choice にした場合、`type="categorical"` + `choices` 形式で保存されることを維持
- 受け入れ条件
  - `metric` 非対応 backend（例: lizyml 0.1.2）でバリデーションエラーを増やさない
  - 既存の Tune 実行導線（`n_trials` / `space`）が不変
  - Search Space `metric` が trial ごとに LightGBM `model.params.metric` へ反映される

#### 20-3. スタイル・アクセシビリティ・回帰確認

- `js/src/widget.css`
  - Tune Settings 用セグメントスタイル（ヒットエリア、active/focus/disabled）を追加
- `tests/`（必要に応じて）
  - metric 切替時に保存値が即時反映されること
  - 初期表示が `METRIC_OPTIONS[task][0]` であること
- 受け入れ条件
  - マウス・キーボード双方で操作できる
  - 既存の Search Space Mode セグメント UI と操作感が揃う

**完了条件:**

- Tune Settings の `metric` がセグメントボタンで表示される
- `Default` 選択肢が存在せず、`METRIC_OPTIONS[task]` から直接選択できる
- schema 非対応時の非表示条件を維持したまま UX が改善される
- Search Space で探索した `metric` が `best_params.metric` として解釈できる

---

### Phase 21: Data タブの Task / CV Strategy チップ選択化（2026-03-13 仕様更新）✅

**目標:** Data タブで頻繁に切り替える `Task` と `CV Strategy` をチップ選択に統一し、選択速度と現在値の視認性を向上する。

#### 21-1. Task 選択 UI のチップ化

- `js/src/tabs/DataTab.tsx`
  - Task 入力をドロップダウンから単一選択チップへ置換
  - 候補は `binary` / `multiclass` / `regression`
  - 自動判定値は初期 active チップとして表示し、`⚡auto` 表示を維持
- 受け入れ条件
  - 1 クリックで Task を切り替えられる
  - 自動判定値と手動変更値が視覚的に区別できる

#### 21-2. CV Strategy 選択 UI のチップ化

- `js/src/tabs/DataTab.tsx`
  - CV Strategy 入力をドロップダウンから単一選択チップへ置換
  - 候補は `split.method` の許可値（`kfold` / `stratified_kfold` / `group_kfold` / `time_series` / `purged_time_series` / `group_time_series`）
  - Strategy 依存表示（group/time 系追加項目）は既存ロジックを維持
- 受け入れ条件
  - Strategy 変更で条件付きフィールド表示が即時更新される
  - 既存の `update_cv` payload 契約が不変

#### 21-3. スタイル・回帰テスト

- `js/src/widget.css`
  - Data タブ用チップ群の折返し・active/focus/disabled スタイルを追加
- `tests/`（必要に応じて）
  - Task/Strategy のチップ選択で `action` payload が正しく送信されることを検証
- 受け入れ条件
  - 狭幅 Notebook でもチップが操作可能
  - キーボード操作（Tab/Enter/Space）で切替できる

**完了条件:**

- Data タブの Task / CV Strategy がチップ選択 UI になっている
- 選択変更時のデータフロー（`set_task` / `update_cv`）が既存契約と一致する
- 可読性と操作速度がドロップダウン方式より改善される

---

### Phase 22: Config 受け渡し契約の確定 + 仕様逸脱検知テスト拡充（計画）

**目標:** UI（Preact）と LizyML backend 間で受け渡す Config 契約を詳細に確定し、契約外フォーマットを自動で検知して Tune/Fit 失敗を未然に防ぐ。

#### 22-1. Config 契約の棚卸しと正規化ポイントの明示

- `BLUEPRINT.md` / 現行コード / LizyML schema を突き合わせ、受け渡し対象キーを「責務別」に棚卸しする
  - Data/Features/Split（Data タブ責務）
  - Model/Training/Evaluation/Calibration（Fit タブ責務）
  - Tuning/Optuna params/space（Tune タブ責務）
- UI 値（Fixed/Range/Choice）から LizyML 受理形式（`type=float/int/categorical`）への変換契約を明記する
- Python 側で補完する既定値（例: `tuning.optuna.params.n_trials`）の適用条件を契約化する
- 受け入れ条件
  - 「どのレイヤーで、どの形に変換・補完するか」が一意に説明できる
  - Tune/Fit 実行前に満たすべき最小構成が明文化される

#### 22-2. 仕様ドキュメントの詳細化（契約テーブル化）

- `BLUEPRINT.md` に Config 契約セクションを追加/拡張し、以下を表形式で確定する
  - path（例: `tuning.optuna.space.<param>`）
  - 型/許容値/必須条件
  - 生成元（UI入力 / Python API / 自動補完）
  - 変換責務（UI / WidgetService / Adapter）
  - 契約違反時の期待エラーコード（`VALIDATION_ERROR` / `BACKEND_ERROR` など）
- 契約更新が変更ゲート対象に該当する場合は `HISTORY.md` Proposal を先行して記録する運用を明記
- 受け入れ条件
  - 受け渡し仕様を読めば、実装せずに payload 正誤判定が可能
  - 既存仕様（P-004/P-005/P-007/P-020）との矛盾がない

#### 22-3. 契約バリデーション強化（実装計画）

- `src/lizyml_widget/adapter.py` / `src/lizyml_widget/service.py` の前処理で契約外入力を早期検知する
  - 旧形式（`mode=range/choice`）の明示的拒否
  - 不正 `type` 値（`range` / `choice` 等）の明示的拒否
  - 必須フィールド欠落時のエラーメッセージ統一
- 受け入れ条件
  - backend 実行前に契約違反を検知できる
  - ユーザーが修正可能なエラーメッセージ（path付き）を返せる

#### 22-4. テスト拡充（仕様逸脱の自動検知）

- `tests/test_frontend_contract.py`
  - Search Space の Mode 切替が `type=float/int/categorical` を保存すること
  - Tune 実行ボタン有効条件と `space` 契約の整合
- `tests/test_widget_api.py`
  - `update_config` 経由で契約外 payload を与えた際の `VALIDATION_ERROR` 検証
  - 既定値補完（`tuning` 欠落時）と契約維持の回帰テスト
- `tests/test_adapter.py`
  - 契約違反ケースをテーブル駆動で追加（legacy mode / invalid type / 欠落項目）
  - 契約準拠ケース（float/int/categorical）が通ることを保証
- `tests/test_e2e.py`
  - UI相当 payload で Fit/Tune 完了まで通る正常系
  - 契約違反 payload で失敗コード・メッセージが期待どおりの異常系
- 受け入れ条件
  - 契約破壊（UI変更・backend更新）を CI で即時検知できる
  - 「失敗を再現してから調査」ではなく「テストで事前検知」へ移行できる

#### 22-5. 回帰運用と完了判定

- CI で契約検証テストを必須ゲートとして扱う
- ドキュメント変更時は契約テスト同時更新をルール化する
- 完了条件
  - Config 契約が BLUEPRINT に固定化され、参照先が一意である
  - 契約逸脱ケースの主要パターンが自動テストで網羅されている
  - UI/Widget/Adapter のどこで壊れてもテストで原因層を特定できる

---

### Phase 23: 入力コントロール統一の追補（2026-03-13 仕様更新）📝

**目標:** Data/Model/Tune の入力操作を「セグメント（単一選択）」「チップ（複数選択）」「数値幅固定 75px」に統一し、UI 一貫性と視認性を高める。

#### 23-1. 数値入力幅を `75px` へ統一

- `js/src/widget.css` / `js/src/components/NumericStepper.tsx`
  - `lzw-stepper` の入力欄を `75px` 固定幅に更新
  - Fit / Data / Tune / Search Space の全 `NumericStepper` で同一幅を適用
- 受け入れ条件
  - 主要数値項目で横幅が統一される
  - 桁数の異なる値でも視認性が維持される

#### 23-2. Data タブの単一選択 UI をセグメントで統一

- `js/src/tabs/DataTab.tsx`
  - Task をセグメントボタン化（`binary` / `multiclass` / `regression`）
  - Cross Validation Strategy をセグメントボタン化（既存候補を維持）
  - Column Settings の Type（Numeric/Categorical）をセグメントボタン化
- 受け入れ条件
  - Task / Strategy / Type の単一選択がすべてセグメントで操作できる
  - `set_task` / `update_cv` / `update_column` の payload 契約が不変

#### 23-3. Model タブの複数選択 UI をチップで統一

- `js/src/tabs/ConfigTab.tsx`
  - Model `metric` をチップボタン（複数選択）に変更
  - Evaluation `metrics` をチップボタン（複数選択）に変更
  - Training `inner_valid` は UI の先頭選択肢を `Default` 表示に統一（保存値は `null`）
- 受け入れ条件
  - metric 系の複数選択 UI がチップボタンに統一される
  - `inner_valid` の初期状態が `Default` として明示される

#### 23-4. Tune Search Space `metric` をチップ複数選択化

- `js/src/components/SearchSpace.tsx`
  - `metric`（Choice モード）の候補選択 UI をチップボタン（複数選択）へ変更
  - `config.choices` への保存形式（string 配列）と `type="categorical"` 契約を維持
- 受け入れ条件
  - Search Space `metric` がチップUIで複数選択できる
  - `tuning.optuna.space.metric` 契約と Tune 実行結果が回帰しない

#### 23-5. テストと回帰検知

- `tests/test_frontend_contract.py` / `tests/test_widget_api.py` / `tests/test_e2e.py`
  - 単一選択（Task/Strategy/Type）がセグメント化されても payload 契約が不変であること
  - 複数選択（Model/Evaluation/SearchSpace metric）がチップ化されても保存値が配列で一致すること
  - `inner_valid=Default` 表示時に `null` 保存となること
  - NumericStepper 幅 `75px` のスタイル契約テスト（CSS 回帰検知）
- 受け入れ条件
  - UI 変更がデータフローを壊さないことをテストで担保できる
  - Notebook 環境差でも表示崩れ・操作退行を検知できる

**完了条件:**

- 数値入力幅が `75px` に統一される
- Data タブの単一選択（Task / Type / Strategy）がセグメント化される
- Model/Tune の複数選択 metric UI がチップボタンに統一される
- `inner_valid` の Default 表示と `null` 保存契約が一致する

---

### Phase 24: Widget / Service 疎結合化（2026-03-13 仕様更新）📝

**目標:** Widget を traitlets / Action / スレッド管理に限定し、config 初期化・実行準備・保存委譲を `WidgetService` に集約して境界を明確化する。

#### 24-1. Widget の backend 直結を解消

- `src/lizyml_widget/widget.py`
  - `LizyWidget.__init__` に `adapter: BackendAdapter | None = None` を追加
  - `adapter` 未指定時のみ `LizyMLAdapter` を既定採用する
- 受け入れ条件
  - `LizyWidget()` の既存呼び出し互換を維持する
  - テストや将来 backend で adapter 注入ができる

#### 24-2. config 初期化 / task 補完 / Tune 補完を Service へ集約

- `src/lizyml_widget/service.py`
  - schema default 展開 + `model.name` / `model.params` 既定値補完を Service の公開メソッドへ移す
  - task 変更時の `objective` / `metric` 補完を Service の公開メソッドへ移す
  - Fit/Tune 実行前の full config 構築と Tune `tuning` 補完を Service の公開メソッドへ移す
- 受け入れ条件
  - Widget が config 補完ロジックを持たない
  - UI / Python API / YAML import の導線で同一の補完規則が使われる

#### 24-3. private 境界越えを除去

- `src/lizyml_widget/widget.py` / `src/lizyml_widget/service.py`
  - `has_data()` / `has_target()` / `save_model()` / config 読込適用 API を Service に追加
  - Widget から `_service._df` / `_service._df_info` / `_service._adapter` 参照を除去する
- 受け入れ条件
  - `widget.py` に Service private 属性参照が残らない
  - Save / Load Config / Fit / Tune の既存挙動が維持される

#### 24-4. テスト整備

- `tests/test_service.py` / `tests/test_widget_api.py`
  - Service の config 初期化・task params 補完・Tune 補完・YAML 適用を単体テスト化する
  - Widget の adapter 注入と Save Model 委譲を検証する
- 受け入れ条件
  - Widget / Service 境界の回帰が自動テストで検知できる
  - private 境界越えが再導入されてもテストで気づける

**完了条件:**

- Widget から Service private 参照が除去される
- config 初期化 / 実行準備 / 保存委譲が Service の公開 API に集約される
- `LizyWidget(adapter=...)` が利用可能になる
- 既存の Fit / Tune / Config I/O テストが通る

---

### Phase 25: Backend Contract 駆動の完全疎結合化（2026-03-13 部分完了 / 要追補）📝

**目標:** backend 固有の option set / parameter catalog / step 値 / default / Search Space 定義を Adapter の `Backend Contract` へ集約し、UI を `backend_contract` + `config` の generic renderer にする。

#### 25-1. 共通型と Adapter Protocol を contract 駆動へ拡張

- `src/lizyml_widget/types.py`
  - `BackendContract`
  - `ConfigPatchOp`
- `src/lizyml_widget/adapter.py`
  - `get_backend_contract()`
  - `initialize_config(task=...)`
  - `apply_config_patch(config, ops, task=...)`
  - `prepare_run_config(config, job_type=..., task=...)`
- 受け入れ条件
  - backend 固有の UI/Config metadata を Adapter だけが所有する
  - WidgetService / UI が backend 固有 default や候補値を再定義しない

#### 25-2. traitlets / Action を patch ベースへ移行

- `src/lizyml_widget/widget.py`
  - `config_schema` traitlet を `backend_contract` traitlet へ置き換える
  - `update_config` action を `patch_config` action へ置き換える
  - canonical config は Python 側でのみ正とし、UI は snapshot を表示する
- 受け入れ条件
  - JS が full config dict を送らなくても設定変更が完結する
  - YAML import / Python API / Notebook UI が同じ canonicalization 経路を通る

#### 25-3. Frontend を contract-driven renderer 化

- `js/src/tabs/ConfigTab.tsx` / `js/src/components/DynForm.tsx` / `js/src/components/SearchSpace.tsx`
  - `backend_contract.ui_schema` からセクション構成・field 表示順・option set・step 値・Search Space catalog を描画する
  - Search Space の `mode=Fixed/Range/Choice` は UI ローカル state に閉じ込め、送信時だけ `type=float/int/categorical` へ正規化する
  - JS から `OBJECTIVE_OPTIONS` / `METRIC_OPTIONS` / LightGBM 固有 parameter catalog / step 定数を除去する
- 受け入れ条件
  - 別 backend を追加しても UI の主要ロジックを書き換えずに描画できる
  - Tune の empty search space 許可などの実行条件が `backend_contract.capabilities` だけで決まる

#### 25-4. Service を orchestration 専任に整理

- `src/lizyml_widget/service.py`
  - Data タブ由来 state（target / task / columns / CV）と canonical config の結合に責務を限定する
  - backend 固有 default / option set / search space catalog / step 値を削除する
  - Fit / Tune 実行前は `prepare_run_config()` を経由して最終 config を組み立てる
- 受け入れ条件
  - Service が backend 固有 constant を持たない
  - 実行可否判定が UI / Python API / YAML import で一致する

#### 25-5. 互換性・テスト・移行

- `tests/test_service.py`
  - canonical config 生成と Data state merge を検証
- `tests/test_widget_api.py`
  - `patch_config` / `backend_contract` traitlet 契約を検証
- `tests/test_frontend_contract.py`
  - contract-driven rendering と Search Space `mode -> type` 正規化を検証
- 既存 `update_config` / `config_schema` 導線は互換期間を設けるか、P-011 に従って一括移行する
- 受け入れ条件
  - backend contract と UI の齟齬を CI で自動検知できる
  - 初期 backend（LizyML）以外を追加しても主要 UI を再設計しなくてよい

**完了条件:**

- UI から backend 固有 option set / parameter catalog / step 定数が除去される
- `backend_contract` traitlet と `patch_config` action で Model/Tune 編集が完結する
- Adapter だけが backend 固有 config metadata を所有する
- canonical config 生成経路が UI / Python API / YAML import で統一される

**監査メモ（2026-03-13）:**

- `25-1` は概ね完了したが、`25-2`〜`25-5` には追補が必要。
- `set_config()` / `load_config()` / `import_yaml` が UI patch と同じ canonicalization 経路を通っていない。
- `training.early_stopping.inner_valid` は監査時点の UI 実装が backend schema とずれており、現在の UI 形式（`"holdout"` / `"fold_0"` など）で `VALIDATION_ERROR` が再現する。
- Tune の empty search space 許可は `backend_contract.capabilities` だけで決まっておらず、frontend が独自判定を持っている。
- frontend / service の backend-specific special case と、CI の検知穴を埋める追補を Phase 26 で扱う。

---

### Phase 26: Canonical Config 経路統一 + `inner_valid` 契約整合化（P-012, 計画）

**目標:** UI / Python API / YAML import の config 導線を単一 canonicalization path に統一し、`inner_valid` の schema 不整合で起きている `VALIDATION_ERROR` を解消する。

**実装状況メモ（2026-03-13 監査追記）:**

- `26-2` と `26-3` は概ね実装済みで、`inner_valid` の object/null 正規化と `error.details` の path/type 出力は確認済み。
- 一方で `26-1` は未完了で、現在の `patch_config` 経路は `unset` 後に `config_version` / `model.name` を再補完せず、`config` traitlet が non-canonical snapshot になりうる。
- `26-4` も未完了で、Tune の empty search space 判定は `backend_contract.capabilities` に寄せられたが、Service に `lgbm` / `objective` / `metric` 固定ロジックが残る。
- `26-5` のテスト追加は進んでいるが、`patch_config` による required field 欠落と public `load_config(path)` 経路は回帰テストが不足している。
- したがって Phase 26 全体の判定は **部分実装（追補要）** とし、下記の不足項目を埋めた時点で完了扱いに更新する。

#### 26-1. 外部 Config 入力の canonicalization を統一

- `src/lizyml_widget/widget.py` / `src/lizyml_widget/service.py` / `src/lizyml_widget/adapter.py`
  - `set_config()` / `load_config()` / `import_yaml` が UI `patch_config` と同じ canonicalization hook を通るよう整理する
  - `config` traitlet には常に canonical config snapshot だけを保持する
  - partial dict / partial YAML 入力でも `config_version` / `model.name` / backend-required default を同一規則で補完する
  - `patch_config` の `set` / `unset` / `merge` のいずれでも、Widget が traitlet へ反映する前に required field / backend default / legacy alias 正規化が再適用されるようにする
  - `build_config()` / `prepare_run_config()` での後追い補完に依存せず、**`config` snapshot 自体** が canonical であることを保証する
- 現在の不足
  - `Widget._handle_patch_config()` → `WidgetService.apply_config_patch()` → `Adapter.apply_config_patch()` の現経路は patch 適用 + 一部正規化に留まり、`model.name` / `config_version` を `unset` すると `get_config()` が non-canonical を返す
  - `set_config()` / `load_config()` / `import_yaml` は canonical 化される一方、UI patch のみ別経路のため「単一 canonicalization path」がまだ達成できていない
- 受け入れ条件
  - `w.set_config({"model": {"params": {}}}).get_config()` が canonical shape を返す
  - `load_config()` / YAML import / UI patch 後の `get_config()` が同じ規則で整形される
  - `patch_config({"ops": [{"op": "unset", "path": "model.name"}]})` 後でも `get_config()["model"]["name"]` が canonical 値を維持する
  - `patch_config({"ops": [{"op": "unset", "path": "config_version"}]})` 後でも `get_config()["config_version"]` が欠落しない

#### 26-2. `training.early_stopping.inner_valid` の契約を backend schema と一致させる

- `BLUEPRINT.md` / `js/src/tabs/ConfigTab.tsx` / `js/src/components/DynForm.tsx` / `src/lizyml_widget/adapter.py`
  - canonical config 上の `inner_valid` を `HoldoutInnerValidConfig | GroupHoldoutInnerValidConfig | TimeHoldoutInnerValidConfig | null` に統一する
  - UI は表示都合で selector state を持てるが、Python に送る payload は object/null へ正規化する
  - `fold_0` など split 表示専用の値は canonical config に入れない
  - 互換期間中は legacy alias (`holdout` / `group_holdout` / `time_holdout`) を Adapter が object へ正規化してもよい
- 受け入れ条件
  - Config Tab で Inner Validation を変更しても `VALIDATION_ERROR` が発生しない
  - Raw Config / YAML export に string alias ではなく canonical object/null が出力される

#### 26-3. Validation failure の診断情報を改善

- `src/lizyml_widget/adapter.py` / `src/lizyml_widget/widget.py`
  - `adapter.validate_config()` が `LizyMLError.__cause__` の `ValidationError.errors()` も読み、field/path/type を返すよう改善する
  - UI / Python API に返す `error.details` で根因 path を追えるようにする
- 受け入れ条件
  - `training.early_stopping.inner_valid` 型不一致時に field path が `error.details` に出る
  - generic な `[CONFIG_INVALID]` だけで止まらず、修正箇所が判別できる

#### 26-4. Phase 25 残課題の追補

- `js/src/tabs/ConfigTab.tsx` / `js/src/components/SearchSpace.tsx` / `src/lizyml_widget/service.py`
  - Tune 実行条件を `backend_contract.capabilities` 起点へ寄せ、frontend 独自判定を削減する
  - backend-specific section/field special case を減らし、generic renderer 化を進める
  - Service に残る `lgbm` / `objective` / `metric` / `auto_num_leaves` 固定ロジックを Adapter へ戻す
- 現在の不足
  - Tune の empty search space 許可可否は frontend 側で `backend_contract.capabilities` を参照するようになったが、Service には `objective` / `metric` の task 依存差し替えと `"lgbm"` の backfill が残っている
  - これらの補完は backend 固有 knowledge であり、WidgetService が「Data タブ由来 state の保持と結合作業に専念する」という責務境界をまだ完全には満たしていない
  - frontend 側も `model` / `training` / `evaluation` などの custom rendering が残るため、Phase 26 では少なくとも backend contract と重複する special case をこれ以上増やさないことを条件にする
- 受け入れ条件
  - empty search space 許可可否が `backend_contract.capabilities` だけで決まる
  - Service が backend 固有 constant を保持しない
  - `src/lizyml_widget/service.py` から backend 名や backend 固有 parameter key 依存の固定ロジックが除去される

#### 26-5. 回帰テストを追加

- `tests/test_widget_api.py` / `tests/test_service.py` / `tests/test_adapter.py` / `tests/test_frontend_contract.py` / `tests/test_e2e.py`
  - `set_config()` / `load_config()` / YAML import の canonicalization 経路統一を検証する
  - `inner_valid` の UI 相当入力と canonical object 出力を検証する
  - `error.details` に nested field path が入ることを検証する
  - `backend_contract.capabilities` による Tune 実行条件を検証する
- 追加で明文化する不足テスト
  - public `load_config(path)` と message-based `import_yaml` の両方が canonical snapshot を返すことを分けて検証する
  - `patch_config` の `unset` により `model.name` / `config_version` が欠落しないことを検証する
  - `save_config()` / `export_yaml` / `raw_config` が canonical object/null を出力することを検証する
- 受け入れ条件
  - 現在の validation failure を CI で再現・検知できる
  - Phase 25 の残課題が再導入されても自動テストで気づける
  - `patch_config` で canonical invariant が破れた場合に CI が必ず落ちる
  - file path API と Notebook custom message API の両方で同一 canonicalization 規約を検知できる

**完了条件:**

- UI / Python API / YAML import に加えて `patch_config` の `set` / `unset` / `merge` 後も `config` traitlet が canonical snapshot を維持する
- `inner_valid` の schema 不整合による `VALIDATION_ERROR` が解消される
- Validation error details に根因 path/type が含まれる
- Service から backend 固有 constant が除去される
- Phase 25 の完了条件と Phase 26 の canonical invariant を CI で継続検証できる

---

### Phase 27: Google Colab 互換のジョブ進捗ポーリング機構（P-018）

**目標:** Google Colab 上でもジョブの進捗表示・完了検知が正常に動作するようにする。既存環境（JupyterLab / VS Code）への影響なし。

#### 27-1. Python 側 poll ハンドラ追加

- `src/lizyml_widget/widget.py`
  - `__init__` に `self.on_msg(self._handle_custom_msg)` を追加
  - `_handle_custom_msg(self, widget, content, buffers)` を実装:
    - `content.type === "poll"` の場合、現在の traitlet 値を `self.send()` で返す
    - 応答ペイロード: `{type: "job_state", status, progress, elapsed_sec, job_type, job_index, error}`
    - `status` が `completed` / `failed` の場合は `fit_summary` / `tune_summary` / `available_plots` も含める
  - 既存の BG スレッド traitlet 書き込みは一切変更しない
- 受け入れ条件
  - `poll` メッセージに対して現在のジョブ状態が `self.send()` で返される
  - メインスレッドで実行される（shell channel 経由）
  - 既存の action ハンドラ / traitlet 同期に影響しない

#### 27-2. JS 側 useJobPolling フック新規作成

- `js/src/hooks/useJobPolling.ts` を新規作成
  - `useJobPolling(model): JobState | null` — ポーリング制御とレスポンス管理
  - `status === "running"` 検出時に 1000ms 間隔で `model.send({type: "poll"})` を開始
  - `model.on("msg:custom")` で `job_state` 応答を受信し、ローカル state に保持
  - JS 側 100ms タイマーで `elapsed_sec` を補間（ポーリング間のスムーズな表示）
  - `status` が `completed` / `failed` になったらポーリング停止 + タイマークリーンアップ
  - `return` 関数でクリーンアップ（`clearInterval` / `model.off`）
- 受け入れ条件
  - ポーリングが `status === "running"` の間のみ動作する
  - 完了後にタイマー / リスナーがリークしない
  - elapsed 表示が 100ms 間隔で滑らかに更新される

#### 27-3. App.tsx で polled state をマージ

- `js/src/App.tsx`
  - `useJobPolling(model)` を呼び出し、polled state を取得
  - polled 値がある場合は traitlet 値を上書き（polled 優先）:
    - `effectiveStatus = polled?.status ?? status`
    - `effectiveProgress = polled?.progress ?? progress`
    - `effectiveElapsedSec = polled?.elapsed_sec ?? elapsedSec`
    - `effectiveFitSummary = polled?.fit_summary ?? fitSummary`（完了時）
    - `effectiveTuneSummary = polled?.tune_summary ?? tuneSummary`（完了時）
    - `effectiveAvailablePlots = polled?.available_plots ?? availablePlots`（完了時）
    - `effectiveError = polled?.error ?? error`
  - 子コンポーネントに effective 値を渡す
- 受け入れ条件
  - JupyterLab: traitlet `change:` が先に到達 → polled は確認程度で冗長な上書き
  - Colab: traitlet `change:` が到達しない → polled state がフォールバックとして UI を更新
  - 両環境で見た目・動作に差がない

#### 27-4. ProgressView に CSS transition 追加

- `js/src/components/ProgressView.tsx` / `js/src/widget.css`
  - プログレスバーの幅変更に `transition: width 0.8s ease-out` を追加
  - ポーリング間隔（1s）の間もバーが滑らかにアニメーション
- 受け入れ条件
  - 進捗バーの変化がカクつかない
  - 既存の JupyterLab 動作で回帰がない

#### 27-5. テスト追加

- `tests/test_widget_api.py`
  - `_handle_custom_msg` が `poll` メッセージに正しい `job_state` を返すことを検証
  - `status` が `running` / `completed` / `failed` の各状態で正しいフィールドが含まれることを検証
  - `poll` 以外のメッセージタイプが無視されることを検証
- 手動検証
  - diagnostic notebook（`debug_colab_traitlet_sync.ipynb`）を Colab で再実行し、LizyWidget の Fit が UI に反映されることを確認
- 受け入れ条件
  - 既存テストが全パス
  - poll ハンドラの回帰テストが追加される

**完了条件:**

- Google Colab 上で Fit/Tune ボタンクリック後に進捗・完了が UI に反映される
- JupyterLab / VS Code での動作に回帰がない
- elapsed 表示が滑らか（JS 補間）
- プログレスバーが CSS transition でアニメーション

---

### Phase 28: ダークモード対応（P-019）

**目標:** JupyterLab / Google Colab / VS Code のダークテーマで Widget が正しく表示されるようにする。

#### 28-1. Widget 固有 CSS 変数の導入

- `js/src/widget.css`
  - `.lzw-root` に Widget 固有 CSS 変数を定義:
    - `--lzw-bg-0` / `--lzw-bg-1` / `--lzw-bg-2` — 背景（base / surface / elevated）
    - `--lzw-fg-0` / `--lzw-fg-1` / `--lzw-fg-2` — 前景（primary / secondary / muted）
    - `--lzw-border` / `--lzw-border-light` — ボーダー
    - `--lzw-accent` / `--lzw-accent-hover` — アクセント
    - `--lzw-success` / `--lzw-warning` / `--lzw-error` — ステータス色
    - `--lzw-input-bg` / `--lzw-input-border` — フォーム入力
  - ライトモードのデフォルト値は現行の色と同一（見た目の変化なし）
  - JupyterLab の `--jp-*` 変数が存在する場合はそちらを優先:
    - `--lzw-bg-0: var(--jp-layout-color0, #fff)`
- 受け入れ条件
  - ライトモードでの見た目が変わらない
  - CSS 変数の命名が体系的で一貫している

#### 28-2. ハードコード色の CSS 変数置換

- `js/src/widget.css`
  - 約 100 箇所のハードコード hex カラー値を `var(--lzw-*)` に置換
  - セクションごとに段階的に置換:
    - Root / Header / Tabs
    - Data タブ（ColumnTable / ChipGroup / SegmentButton）
    - Config タブ（DynForm / SearchSpace / NumericStepper）
    - Results タブ（ScoreTable / ProgressView / PlotViewer）
    - 共通コンポーネント（Button / Toggle / Modal）
- 受け入れ条件
  - ライトモードでピクセルパーフェクト（色の変化なし）
  - 全ハードコード色が CSS 変数に置換される

#### 28-3. ダークモードメディアクエリ追加

- `js/src/widget.css`
  - `@media (prefers-color-scheme: dark)` ブロックで `--lzw-*` 変数をダーク値に上書き
  - JupyterLab: `--jp-*` 変数が存在すればそちらが優先される（`var(--jp-*, var(--lzw-*))` フォールバック）
  - Colab / `--jp-*` 未提供環境: Widget 固有のダーク変数にフォールバック
  - ダーク色は JupyterLab Dark テーマの色調に合わせて設計
- 受け入れ条件
  - JupyterLab ダークテーマで自然な見た目
  - Colab ダークモードで Widget が浮かない
  - VS Code ダークテーマで統合感がある

#### 28-4. Plotly プロットのテーマ追従

- `js/src/hooks/usePlot.ts` または Plotly レンダリング部
  - `window.matchMedia("(prefers-color-scheme: dark)")` でダークモード検出
  - ダーク時に Plotly の `layout.template` を `plotly_dark` に設定
  - `layout.paper_bgcolor` / `layout.plot_bgcolor` を Widget の背景色に合わせる
  - テーマ切替時のリアクティブ更新（`matchMedia.addEventListener("change", ...)`)
- 受け入れ条件
  - プロットの背景・テキスト・グリッド色がダークモードに追従
  - ライトモード ↔ ダークモードの動的切替に対応

#### 28-5. WCAG コントラスト比の自動検証テスト

- `tests/test_css_contrast.py` を新規作成
  - `widget.css` をパースし、`.lzw-root` と `@media (prefers-color-scheme: dark)` ブロックから `--lzw-*` 変数の定義値を抽出
  - 前景/背景のペア定義テーブルを用意:
    ```
    ("--lzw-fg-0", "--lzw-bg-0"),   # 本文テキスト on 基本背景
    ("--lzw-fg-1", "--lzw-bg-0"),   # 副テキスト on 基本背景
    ("--lzw-fg-2", "--lzw-bg-0"),   # 補助テキスト on 基本背景
    ("--lzw-fg-0", "--lzw-bg-1"),   # 本文テキスト on サーフェス背景
    ("--lzw-fg-0", "--lzw-bg-2"),   # 本文テキスト on 高位背景
    ("--lzw-fg-0", "--lzw-input-bg"),  # テキスト on 入力欄
    ("--lzw-accent", "--lzw-bg-0"),    # アクセント色 on 基本背景
    ("--lzw-error", "--lzw-bg-0"),     # エラー色 on 基本背景
    ("--lzw-success", "--lzw-bg-0"),   # 成功色 on 基本背景
    ("--lzw-warning", "--lzw-bg-0"),   # 警告色 on 基本背景
    # ... ボタン・バッジ等のペアも追加
    ```
  - WCAG 2.1 コントラスト比計算を実装（相対輝度 → コントラスト比）:
    - AA 基準: 通常テキスト ≥ 4.5:1、大テキスト/UI 要素 ≥ 3.0:1
  - ライトモード・ダークモード両方のペアに対してコントラスト比を検証
  - 違反があれば変数名・実際のコントラスト比・要求値を含む明確なエラーメッセージを出力
- テストの位置づけ
  - CI で自動実行される（`uv run pytest` の対象に含まれる）
  - CSS 変数を変更するたびに、見えにくい色の組み合わせが導入されないことを保証
  - ブラウザ不要（CSS テキストの静的解析のみ）
- 受け入れ条件
  - ライトモード・ダークモードの全前景/背景ペアが WCAG AA 基準を満たす
  - コントラスト比不足時にテストが失敗し、具体的な変数名と数値を報告する

#### 28-6. ハードコード色の残存チェック（CI lint）

- `tests/test_css_contrast.py` に追加
  - `widget.css` の `--lzw-*` 変数定義行とコメント行を除外した上で、残存するハードコード hex カラー（`#xxx` / `#xxxxxx` / `rgb(...)` / `rgba(...)`）を検出
  - CSS 変数への置換漏れを CI で自動検知
- 受け入れ条件
  - ハードコード色がゼロ（変数定義行を除く）

#### 28-7. 手動視覚確認

- 6 環境での動作確認
  - JupyterLab ライト / ダーク
  - Colab ライト / ダーク
  - VS Code ライト / ダーク
  - 全タブ（Data / Config / Results）で確認
- 受け入れ条件
  - 6 環境で視覚的な不具合がない

**完了条件:**

- ハードコードカラー約 100 箇所が CSS 変数に置換される
- `@media (prefers-color-scheme: dark)` でダークモード変数が定義される
- JupyterLab / Colab / VS Code の各ダークテーマで Widget が適切に表示される
- Plotly プロットがダーク/ライトテーマに追従する
- 全前景/背景ペアが WCAG AA コントラスト比基準（4.5:1 / 3.0:1）を CI で自動検証される
- ハードコード色の残存が CI で自動検出される
