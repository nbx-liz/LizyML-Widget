## LizyML-Widget 仕様変更履歴

### P-001: `set_task` アクション追加

- **日付**: 2026-03-10
- **ステータス**: Approved（2026-03-10 承認）
- **背景**: BLUEPRINT §5.2 では Task を「自動判定結果を初期値とするドロップダウン（変更可能）」と定義しているが、§3.6 アクション一覧に `set_task` が含まれていない。ユーザーが Task を手動変更するためのアクションが必要。
- **提案内容**:
  - 新規アクション `set_task` を追加（payload: `{"task": "binary" | "multiclass" | "regression"}`）
  - `WidgetService.set_task(task)` メソッドを追加（task 更新 + CV strategy デフォルト再設定）
  - `df_info` に `auto_task` フィールドを追加（自動判定値を保持し、UI で「⚡auto」表示の判定に使用）
- **影響範囲**: `action` traitlet のペイロード種別追加、`df_info` Dict 内フィールド追加、Service メソッド追加
- **BLUEPRINT 更新**: §3.6 アクション一覧に `set_task` を追記、§3.5 `df_info` に `auto_task` を追記

### P-003: チュートリアル向け Python API 拡張（`set_target` / `fit` / `tune` / 読み取りプロパティ）

- **日付**: 2026-03-11
- **ステータス**: Approved（2026-03-11 承認）
- **背景**:
  - チュートリアル Notebook（`notebooks/tutorial.ipynb`）は `w.set_target(col)` / `w.fit()` / `w.tune()` / `w.task` / `w.cv_method` / `w.cv_n_splits` / `w.df_shape` / `w.df_columns` を呼び出しているが、これらは `LizyWidget` に実装されていない。
  - BLUEPRINT §4.1 の Python API 仕様にもこれらが記載されておらず、PLAN.md にも実装タスクが挙がっていない。
  - `nbconvert --execute` を使ったチュートリアル実行検証で `AttributeError` が発生し、Fit 完了が確認できない状態。
- **提案内容**:
  - `set_target(col: str) -> LizyWidget` — `_service.set_target()` を呼び `df_info` / `status` を更新するパブリックメソッド。`self` を返しチェーン可能。
  - `fit(*, timeout: float | None = None) -> LizyWidget` — `_run_job("fit")` をバックグラウンドで起動し、`threading.Event` で完了（`status == "completed"` or `"failed"`）を待つブロッキングメソッド。失敗時は `RuntimeError` を raise。`self` 返却。
  - `tune(*, timeout: float | None = None) -> LizyWidget` — `fit()` と同パターンで `_run_job("tune")` を待機。
  - `task: str | None` プロパティ — `df_info.get("task")` を返す読み取り専用プロパティ。
  - `cv_method: str` プロパティ — `df_info.get("cv", {}).get("strategy", "kfold")` を返す。
  - `cv_n_splits: int` プロパティ — `df_info.get("cv", {}).get("n_splits", 5)` を返す。
  - `df_shape: list[int]` プロパティ — `df_info.get("shape", [])` を返す。
  - `df_columns: list[dict]` プロパティ — `df_info.get("columns", [])` を返す。
- **影響範囲**:
  - `src/lizyml_widget/widget.py` — メソッド・プロパティの追加
  - BLUEPRINT.md §4.1 — Python API 仕様への追記
  - PLAN.md — Phase 10-7（E2E テスト）の前提タスクとして記録
- **BLUEPRINT 更新**: §4.1 の LizyWidget クラス定義に上記メソッド・プロパティを追記

---

### P-002: Data/Model タブ設定要件の明文化（LizyML schema 準拠）

- **日付**: 2026-03-10
- **ステータス**: Approved（2026-03-10 承認）
- **背景**:
  - BLUEPRINT §5.2/§5.3 は Data/Config UI の概要説明はあるが、LizyML 設定キーと初期値の対応が不十分。
  - 特に Data → Config 反映キーに `data.task` / `split.strategy` / `split.group_column` が記載されており、LizyML 側の正（`task` / `split.method` / `data.group_col`）と不一致。
  - ユーザー要求として、Data タブと Model タブ（旧 Config タブ）の要件を、LightGBM と Tuning を含めて初期値付きで明確化する必要がある。
- **提案内容**:
  - BLUEPRINT §5.1 のタブラベルを `Model` 表記へ更新し、§5.3 を「Model タブ（旧 Config タブ）」として定義する。
  - BLUEPRINT §5.2 に Data タブ要件表を追加し、LizyML キー・初期値・表示条件を明記する。
  - Data タブの反映先キーを LizyML schema 準拠に修正する（`task` / `split.method` / `data.group_col` / `data.time_col` など）。
  - BLUEPRINT §5.3 に Model タブ要件表を追加し、LightGBM（Fit）と Tuning（Optuna）の初期値を明記する。
- **影響範囲**:
  - Widget タブ間の設定データフロー仕様（ドキュメント定義）
  - Data/Model タブの UI 要件（表示項目、デフォルト値、保存キー）
- **BLUEPRINT 更新**:
  - §5.1 タブバー表記
  - §5.2 Data タブ要件（初期値付き）
  - §5.3 Model タブ要件（LightGBM + Tuning 初期値付き）

---

### P-004: Tune 起動時の `tuning` デフォルト補完 + SearchSpace 契約修正

- **日付**: 2026-03-12
- **ステータス**: Approved（2026-03-12 承認）
- **背景**:
  - Tune が Python API / Widget UI どちらの導線でも失敗する 4 つの再現パターンを確認。
  - R1: `w.tune()` で `tuning` 未設定 → `CONFIG_INVALID`。
  - R2: SearchSpace で Range/Choice 設定 → `Unknown search space type ''`（UI が `mode` 形式で保存するが LizyML は `type` 形式を要求）。
  - R3: Tune 本体成功後に `evaluate_table()` / `split_summary()` で `MODEL_NOT_FIT`。
  - R4: Tune-only 後の `available_plots` が Fit 依存プロットを含み取得時にエラー。
- **提案内容**:
  - `_run_job("tune")` 実行時に `config.tuning` が欠落 / 不完全なら最小有効構成 `{"optuna": {"params": {"n_trials": 50}, "space": {}}}` を自動補完する（R1）。
  - `SearchSpace.tsx` の `handleUpdate` で `ParamConfig`（UI 内部状態 `mode` ベース）を LizyML 契約形式（`type` ベース）に変換してから `onChange` を呼ぶ（R2）。
    - Range → `{type: "float"/"int", low, high, log}`
    - Choice → `{type: "categorical", choices: [...]}`
    - Fixed → key を `space` から削除（現行通り）
  - Tune 後の `evaluate_table()` / `split_summary()` 呼び出しを try/except でガードし、Tune-only 時の `MODEL_NOT_FIT` で Tune 成功を失敗にしない（R3）。
  - `adapter.available_plots()` に `is_fitted` 判定を追加し、Fit 依存プロット（`learning-curve`, `oof-distribution`, `feature-importance` 等）を Fit 済み時のみ返す（R4）。
  - `adapter.validate_config()` に旧形式（`mode` あり `type` なし）の防御バリデーションを追加。
- **影響範囲**:
  - `src/lizyml_widget/widget.py` — `_run_job()` の tuning 補完、`_job_worker()` の try/except ガード
  - `src/lizyml_widget/adapter.py` — `available_plots()` の Fit 状態判定、`validate_config()` の space 検証
  - `js/src/components/SearchSpace.tsx` — `handleUpdate` の出力形式変換
  - `js/src/tabs/ConfigTab.tsx` — `hasSearchParam` の条件変更
- **BLUEPRINT 更新**: Tune ボタン有効条件の明文化（UI は Range/Choice 1件必須、Python API は `space={}` 許容）
- **Decision**: 2026-03-12 実装完了。全 4 パターン（R1〜R4）を修正しテストで回帰検知可能。

---

### P-005: Apply to Fit で Tune 実行時設定をフル同期

- **日付**: 2026-03-12
- **ステータス**: Approved（2026-03-12 承認）
- **背景**:
  - 現行の `apply_best_params` は Best Params を `model.params` にマージする中心実装であり、Tune 実行時に使った他の設定（training/evaluation/calibration 等）と乖離する場合がある。
  - ユーザー要件として、Tune 完了後に [Apply to Fit ▸] を押した際、Fit 画面のパラメータを Tune 実行時と同一状態に揃える必要がある。
- **提案内容**:
  - Tune 実行開始時の有効 config をスナップショットとして保持する（Widget/Service 内部状態）。
  - `apply_best_params` 実行時は以下の順で適用する:
    1. Tune 実行時 config スナップショットを復元
    2. `best_params` を `model.params` に上書き
    3. Model タブ Fit サブタブへ切り替える
  - これにより、Fit 画面の全パラメータ（model/training/evaluation/calibration/output_dir 等）が Tune 実行時設定と一致する。
- **影響範囲**:
  - Widget タブ間データフロー（Results → Model Fit）
  - `apply_best_params` action の意味論
  - Fit 画面で表示される config の再現性
- **BLUEPRINT 更新**:
  - §3.6 `apply_best_params` の説明を「Best Params マージ」から「Tune 実行時設定復元 + Best Params 適用」に更新
  - §5.4 Tune 完了の Apply to Fit 動作仕様を更新
- **Decision**: 2026-03-12 に仕様として採用。PLAN に実装フェーズを追加。

---

### P-006: Tune Settings `metric` UI をセグメントボタン化

- **日付**: 2026-03-12
- **ステータス**: Approved（2026-03-12 承認）
- **背景**:
  - Tune Settings の `metric` がプルダウンメニューだと、候補比較と現在値の視認性が低く操作コストが高い。
  - Search Space の Mode はすでにセグメントボタン方針であり、Tune Settings 側も同じ操作体系へ揃えることで学習コストを下げられる。
- **提案内容**:
  - Tune Settings の `metric` 入力 UI を「セレクト」から「セグメントボタン」に変更する。
  - 候補は `Default`（`null`）+ `METRIC_OPTIONS[task]` の 1 つ選択とする。
  - `Default` 選択時は現行仕様どおり `tuning.optuna.params.metric = null` を保持する。
- **影響範囲**:
  - UI 操作方式（Tune Settings）
  - ドキュメント（BLUEPRINT §5.3 / PLAN）
- **BLUEPRINT 更新**:
  - §5.3 Tune サブタブの ASCII 図をセグメント表現へ更新
  - Tune サブタブ要件の `Metric` 備考を「セグメントボタン」へ更新
- **Decision**: 2026-03-12 に仕様として採用。PLAN に実装フェーズを追加。

---

### P-007: Data/Tune UI 操作系の一括改善（チップ化・Grid化・数値入力可読性）

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **背景**:
  - Data タブの Task / CV Strategy がドロップダウン中心で、操作回数が多く現在値比較もしづらい。
  - Column Settings / Search Space はテーブル幅制御の限界で、環境によって間延びや可読性低下が残る。
  - Tune Settings `metric` の `Default` は意図が伝わりづらく、選択UIの明確性を下げる。
  - Search Space の Range 入力と数値欄幅に、押しにくさ・桁欠けの UX 問題がある。
- **提案内容**:
  - Task をドロップダウンからチップ選択へ変更する（`binary` / `multiclass` / `regression`）。
  - Cross Validation の Strategy をドロップダウンからチップ選択へ変更する。
  - Column Settings / Search Space を `<table>` から CSS Grid に置換し、`minmax()` で列幅を自動調整する。
  - Tune Settings `metric` から `Default` を削除し、task 別 metric のみをセグメント表示する。
  - Search Space の Range（`low` / `high`）を大型 `- / +` ステッパーへ統一する。
  - 数値入力欄の最小幅を拡張し、桁欠けを防止する（例: `min-width: 8ch`）。
- **影響範囲**:
  - Data タブ操作 UI（Task / CV Strategy）
  - Column Settings / Search Space のレイアウト実装
  - Tune Settings `metric` の選択仕様
  - 数値入力コンポーネントの表示契約
- **BLUEPRINT 更新**:
  - §5.2 Target / Task と Cross Validation をチップ選択仕様へ更新
  - §5.2 / §5.3 の Column Settings / Search Space レイアウトを CSS Grid + `minmax()` 仕様へ更新
  - §5.3 Tune `metric` から `Default` を除去し、初期値を `METRIC_OPTIONS[task][0]` に更新
  - 数値入力欄の `min-width` 仕様を追記
- **Decision**: 2026-03-13 に仕様として採用。PLAN の Phase 17/18/20 を更新し、Task/CV チップ化の実装フェーズを追加。

---

### P-008: Search Space `metric` の意味論明確化（LightGBM パラメータ）

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **背景**:
  - Tune 設定には `tuning.optuna.params.metric` と `tuning.optuna.space.metric` の 2 系統があり、用途が混同されやすい。
  - ユーザー要件として、Search Space 内の `metric` が LightGBM パラメータとして扱われることを仕様上で明示する必要がある。
- **提案内容**:
  - `tuning.optuna.space.metric` は **LightGBM `model.params.metric` の探索軸**であることを明記する。
  - Choice で選ばれた候補が trial ごとに `model.params.metric` へ適用されることを明記する。
  - Tune 完了時の `best_params.metric` は LightGBM パラメータ値として解釈することを明記する。
- **影響範囲**:
  - BLUEPRINT §5.3（Tune/Search Space）
  - BLUEPRINT §5.4（Tune 完了時の Best Params 解釈）
- **BLUEPRINT 更新**:
  - `tuning.optuna.params.metric` と `tuning.optuna.space.metric` の役割分離注記を追加
  - Search Space パラメータ表の `metric` 行に「`model.params.metric` として適用」を追記
  - Tune 完了セクションに `best_params.metric` の意味を追記
- **Decision**: 2026-03-13 に仕様として採用。PLAN の Tune 関連フェーズへ明確化タスクを追記。

---

### P-009: 入力コントロール統一の追補（75px固定 + セグメント/チップ化 + Inner Valid Default）

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **背景**:
  - 数値入力欄は現在「最小幅」ベースの記述であり、画面やフォント差で見え方がぶれる。
  - Data タブ / Model タブ / Search Space にチェックボックス・ドロップダウン混在が残っており、操作体系を統一しきれていない。
  - Training の `inner_valid` で `null` 表示が直接見えると、ユーザーにとって初期状態の意味が分かりづらい。
- **提案内容**:
  - `lzw-stepper` の数値入力欄幅を `75px` 固定に統一する。
  - Data タブ:
    - Task をセグメントボタンに統一する。
    - Column Settings の Type（Numeric/Categorical）をセグメントボタンに変更する。
    - Cross Validation の Strategy をセグメントボタンに統一する。
  - Model タブ:
    - Model `metric` をチップボタン（複数選択）に変更する。
    - Evaluation `metrics` をチップボタン（複数選択）に変更する。
    - Training `inner_valid` は UI 表示を `Default` とし、保存値は `null` を維持する。
  - Tune Search Space:
    - `metric` の Choice UI をチップボタン（複数選択）に変更する。
- **影響範囲**:
  - BLUEPRINT §5.2（Data タブ UI）
  - BLUEPRINT §5.3（Model/Tune UI、Search Space UI、数値入力仕様）
  - PLAN.md（未実装項目の実装フェーズ追加）
- **BLUEPRINT 更新**:
  - Task / CV Strategy / Type の UI 表記をセグメントボタンへ更新
  - metric 系（Model/Evaluation/Search Space）の UI 表記をチップボタンへ更新
  - `inner_valid` の `Default` 表示ルールを追記
  - 数値入力欄幅を `75px` 固定へ更新
- **Decision**: 2026-03-13 に仕様として採用。PLAN に実装フェーズを追加。

---

### P-010: Widget / Service 境界の疎結合化（config 初期化・実行準備の Service 集約）

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **関連**: BLUEPRINT.md §3.2, §4.1, §10.1
- **背景**:
  - 現行実装では `LizyWidget` が `LizyMLAdapter` を直接生成し、`_service._df` / `_service._df_info` / `_service._adapter` へ private アクセスしている。
  - config 初期化・task 依存 params 補完・Tune 用 `tuning` 補完・YAML 読込適用が Widget / Service / UI に分散しており、導線ごとの挙動差（UI と Python API のズレ）が起きやすい。
  - アーキテクチャ上は Widget が traitlets / Action / スレッド管理に専念し、Service が config 正規化と実行前提の責務を持つほうが疎結合で保守しやすい。
- **提案内容**:
  - `LizyWidget` のコンストラクタは `adapter: BackendAdapter | None = None` を受け取り、未指定時のみ `LizyMLAdapter` を使用する。
  - `WidgetService` に以下の公開責務を追加する:
    - config 初期化（schema default 展開 + `model.name` / `model.params` 既定値補完）
    - task 依存 params の補完
    - YAML / dict 読込時の `data` / `split` 適用
    - Fit / Tune 実行前の full config 構築と Tune 既定値補完
    - `has_data()` / `has_target()` による実行前提判定
    - モデル保存の委譲
  - `LizyWidget` は Service の公開メソッド経由でのみ状態を参照・更新し、Service の private 属性へ直接アクセスしない。
  - 既存の `BackendAdapter` Protocol は今回変更しない。backend capability metadata の一般化は別 Proposal で扱う。
- **影響範囲**:
  - Python API（`LizyWidget.__init__` の任意 adapter 注入）
  - Widget / Service 間の内部データフロー
  - BLUEPRINT のレイヤ責務・Config 契約
- **互換性**:
  - 既存の `LizyWidget()` 呼び出しはそのまま有効。
  - `adapter` 引数は追加のみで、既存利用者への破壊的変更はない。
- **代替案**:
  - `BackendAdapter` Protocol を拡張して objective / metric / tunable param catalog も adapter から供給する案。
  - 今回は変更面積が大きく、UI 仕様と Protocol の再設計が必要になるため見送る。
- **受け入れ条件**:
  - `widget.py` から `_service._df` / `_service._df_info` / `_service._adapter` 参照が除去される。
  - `load()` / `load_config()` / `fit()` / `tune()` / `save_model()` が従来どおり動作する。
  - `WidgetService` に追加した公開メソッドを単体テストで検証する。
- **Decision**: 2026-03-13 に仕様として採用。BLUEPRINT / PLAN / CLAUDE / AGENTS と実装を同期する。
- **Migration**:
  - 既存コードの移行は不要。
  - テストや外部コードで `LizyWidget` に別 backend を差し込みたい場合のみ `LizyWidget(adapter=...)` を利用できる。

---

### P-011: Backend Contract 駆動 UI / Patch ベース更新による完全疎結合設計

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **関連**: BLUEPRINT.md §3.2, §3.3, §3.4, §3.6, §5.3, §6.3
- **背景**:
  - P-010 により Widget から Service private 参照は除去したが、UI と Service には依然として LizyML / LightGBM 固有の option set・parameter catalog・step 値・search space 行定義が残る。
  - `ConfigTab.tsx` / `SearchSpace.tsx` が objective / metric 候補や tunable param 一覧を保持している限り、backend を差し替えるたびに JS と Service の両方を修正する必要があり、「Adapter で吸収する」原則を満たせない。
  - UI が `update_config` で full config dict を送る方式は、backend 固有の path / default / 補完規則を UI に漏らしやすい。
- **提案内容**:
  - `BackendAdapter` Protocol を拡張し、runtime API に加えて **Backend Contract** を返す。
    - `get_backend_contract()` — `config_schema`、`ui_schema`、`capabilities` をまとめて返す
    - `initialize_config()` — backend 固有 default を含む canonical config を生成
    - `apply_config_patch()` — UI からの patch operation を canonical config へ適用する
    - `prepare_run_config()` — Fit/Tune 実行前の backend 固有補完を行う
  - UI は backend 固有の option list / parameter catalog / step 値を保持しない。
    - UI は `backend_contract.ui_schema` を読んでフォーム・Search Space・選択肢・表示条件を構築する
    - UI から Python への編集イベントは full config ではなく `patch_config` action に統一する
  - `WidgetService` は backend 固有定数を持たず、Data タブ由来状態と Adapter Contract を仲介する。
    - 自動判定・Column/CV 管理・Data/Features/Split 生成は引き続き Service の責務
    - Model/Tune 固有の default / option / patch 意味論は Adapter 側へ移す
  - `config_schema` traitlet は廃止し、`backend_contract` traitlet に統合する。
  - `update_config` action は廃止し、UI 編集は `patch_config` のみを使用する。
- **影響範囲**:
  - `BackendAdapter` Protocol
  - 共通型（`BackendContract`, `ConfigPatchOp` など）
  - traitlets (`backend_contract` 追加 / `config_schema` 廃止)
  - Action 契約（`patch_config` 追加 / `update_config` 廃止）
  - Config / Search Space UI の描画方式
- **互換性**:
  - 既存の Python API (`set_config`, `load_config`, `fit`, `tune`) は維持する。
  - JS UI 実装は破壊的変更となるため、Phase 25 で段階移行する。
  - 移行期間中は Python 側で `update_config` を後方互換として受理してもよいが、仕様上の正は `patch_config` とする。
- **代替案**:
  - 現行の `config_schema` + JS hardcode を維持し、backend 追加時にフロントを個別修正する案。
  - `WidgetService` に backend 別 catalog を持たせる案。
  - いずれも UI / Service に backend 固有知識が残るため、不採用。
- **受け入れ条件**:
  - BLUEPRINT 上、UI は backend 固有 option set / parameter catalog を保持しないと明記される。
  - Adapter Protocol に backend contract / patch / config lifecycle hook が定義される。
  - traitlets と Action 契約が `backend_contract` / `patch_config` ベースに更新される。
  - PLAN に完全疎結合化の実装フェーズが追加される。
- **Decision**: 2026-03-13 に仕様として採用。ドキュメントを更新し、実装は次フェーズで行う。
- **Migration**:
  - JS 実装は `update_config({config})` から `patch_config({ops})` へ移行する。
- `config_schema` traitlet 依存のコードは `backend_contract.config_schema` を参照する。
- objective / metric / search space catalog は frontend 定数から削除し、adapter が返す `ui_schema` へ移す。

---

### P-012: Canonical Config 経路統一 / `inner_valid` 契約整合化 / Validation 診断改善

- **日付**: 2026-03-13
- **ステータス**: Approved（2026-03-13 承認）
- **関連**: BLUEPRINT.md §3.3, §4, §5.3, §6.3, PLAN.md Phase 25 / Phase 26
- **背景**:
  - Phase 25 の要件監査により、`backend_contract` / `patch_config` の導入自体は完了した一方、`set_config()` / `load_config()` / YAML import が UI patch と同じ canonicalization 経路を通っていないことを確認した。
  - `ConfigTab.tsx` の `training.early_stopping.inner_valid` は `holdout` / `fold_0` などの文字列を保存するが、LizyML schema は `HoldoutInnerValidConfig | GroupHoldoutInnerValidConfig | TimeHoldoutInnerValidConfig | null` を要求する。
  - 実行再現では `inner_valid="holdout"` を含む config が `VALIDATION_ERROR` で停止したが、`adapter.validate_config()` は外側の `LizyMLError` しか読まず、根因の field/path を UI に返せていない。
- **提案内容**:
  - UI patch / `set_config()` / `load_config()` / `import_yaml` の全導線を、Adapter 主導の同一 canonicalization 経路へ統一する。
    - `config` traitlet は常に canonical config の snapshot とし、required field / backend default / legacy alias 正規化後の値だけを保持する。
    - 外部入力が partial dict / partial YAML でも、backend 必須フィールド（`config_version`, `model.name` など）は canonicalization の中で補完する。
  - `training.early_stopping.inner_valid` の canonical 型は backend schema を正とし、**object または null** に統一する。
    - UI は表示都合で短いラベルや selector state を持てるが、Python に送る `patch_config` payload は `{method: ...}` を含む object または `null` に正規化する。
    - 互換期間中は legacy alias (`"holdout"`, `"group_holdout"`, `"time_holdout"`) のみ Adapter で object へ正規化してもよいが、`"fold_0"` 等の表示専用値は canonical config に入れない。
  - `adapter.validate_config()` は `LizyMLError.__cause__` にぶら下がる `ValidationError.errors()` も参照し、`field` / `message` / `type` を UI へ返す。
- **影響範囲**:
  - Python API（`set_config`, `load_config`）
  - Widget / Service / Adapter の config lifecycle
  - Config Tab の `inner_valid` 編集 UI
  - Validation エラー表示と回帰テスト
- **互換性**:
  - 既存の Python API 名（`set_config`, `load_config`, `fit`, `tune`）は維持する。
  - `get_config()` / Raw Config / YAML export が返す値は、これまでより canonical 寄りになる。
  - 既存ユーザーが legacy string alias を渡した場合は、互換期間中に限り canonical object へ正規化する。
- **代替案**:
  - `service.build_config()` / `prepare_run_config()` でだけ補完を続け、`config` traitlet は非 canonical のまま許容する案。
  - `inner_valid` の UI だけを個別修正し、Python API / YAML import は現状維持とする案。
  - いずれも Phase 25 の「単一 canonicalization 経路」と「Python 側 canonical config」要件を満たせないため不採用。
- **受け入れ条件**:
  - `set_config()` / `load_config()` / `import_yaml` / `patch_config` 後の `config` traitlet が同一規則で canonical 化される。
  - `training.early_stopping.inner_valid` を UI から変更しても `VALIDATION_ERROR` が再現しない。
  - Validation エラー詳細に `training.early_stopping.inner_valid` の path と型不一致理由が含まれる。
  - 回帰テストが canonicalization 経路統一と `inner_valid` 契約逸脱の両方を検知できる。
- **Decision**: 2026-03-13 に修正方針として採用。BLUEPRINT / PLAN に残課題と追補フェーズを反映する。
- **Migration**:
  - 既存の UI / Python API / YAML 利用者は API 名の変更なし。
  - `inner_valid` の legacy string alias は互換期間中に canonical object へ正規化し、将来的に非推奨化する。
  - `config` snapshot の shape が canonical object ベースに揃うため、非 canonical 値を前提にしたデバッグコードは読み替えが必要。

---

### A-2026-03-13: Phase 25 監査記録（部分完了 / Config validation failure 追跡）

- **日付**: 2026-03-13
- **種別**: Audit Finding（要件監査の記録）
- **背景**:
  - `PLAN.md` では Phase 25 を 2026-03-13 完了扱いとしていたが、コード監査と実行再現では完了条件の一部が未達だった。
  - あわせて、Config Tab 編集後に `VALIDATION_ERROR` が残る経路を調査した。
- **確認した問題点**:
  - `set_config()` / `load_config()` / `_apply_loaded_config()` が Adapter canonicalization hook を通らず、`config` traitlet が非 canonical snapshot を保持しうる。
  - `ConfigTab.tsx` の `inner_valid` は `fold_0` / `holdout` などの string を保存するが、LizyML schema は object/null を要求するため validation が失敗する。
  - 再現確認では `training.early_stopping.inner_valid = "holdout"` を含む config で `VALIDATION_ERROR` を確認し、根因は `ValidationError: Input should be a valid dictionary...` だった。
  - `adapter.validate_config()` は `LizyMLError.__cause__` の詳細を吸い上げないため、UI には generic な `[CONFIG_INVALID]` しか表示されない。
  - Phase 25 の残課題として、`backend_contract.capabilities` 未使用の Tune 実行条件判定と、frontend / service の backend-specific special case も残っている。
- **影響範囲**:
  - Config Tab の Training セクション
  - Python API / YAML import と Notebook UI の整合性
  - Validation failure 時のデバッグ容易性
  - Phase 25 完了判定の信頼性
- **対応先**:
  - HISTORY.md P-012
  - PLAN.md Phase 25 残課題メモ / Phase 26 追補

---

### A-2026-03-13: Phase 26 実装監査記録（部分実装 / canonical snapshot 不変条件の追補）

- **日付**: 2026-03-13
- **種別**: Audit Finding（要件監査の記録）
- **背景**:
  - Phase 26 の実装状況を監査し、P-012 で定義した受け入れ条件が current working tree でどこまで満たされているかを確認した。
  - `inner_valid` 契約整合化と Validation 診断改善は概ね完了していた一方、canonical config の不変条件と Service 疎結合化には残課題があることを確認した。
- **確認した問題点**:
  - `patch_config` の `unset` 後に `config_version` / `model.name` が再補完されず、`config` traitlet が non-canonical snapshot を保持しうる。
  - `set_config()` / `load_config()` / `import_yaml` は canonical 化されるが、UI patch だけ別の保証水準になっており、「単一 canonicalization path」が未達。
  - Service には `lgbm` / `objective` / `metric` 固定ロジックが残っており、backend 固有 knowledge を Adapter へ集約する Phase 26-4 の完了条件に未達。
  - 回帰テストは増えているが、public `load_config(path)` と `patch_config unset` canonical invariant、`save_config()` / `export_yaml` / `raw_config` の canonical 出力が CI で十分に固定化されていない。
- **影響範囲**:
  - Config Tab の patch 適用後 snapshot の信頼性
  - Python API / Notebook UI / YAML I/O の canonical config 一貫性
  - Service / Adapter の責務境界
  - CI による Phase 26 完了判定の信頼性
- **対応先**:
  - PLAN.md Phase 26 監査追記
  - `26-1` / `26-4` / `26-5` の不足項目明文化

---

### A-2026-03-12: Fit 実行失敗（`model.name` 欠落）監査記録

- **日付**: 2026-03-12
- **種別**: Audit Finding（要件監査の記録）
- **背景**:
  - `w.load(df, target=...)` 直後に `fit` を実行すると `VALIDATION_ERROR` で停止し、学習が開始されない事象を確認。
  - 実装監査（Widget/Service/Adapter/UI）と実行再現により、初期 config の `model.name` 欠落が主因であることを特定。
- **確認した問題点**:
  - 初期 config 生成で `model.name` が設定されない場合がある（`oneOf + discriminator` な schema で `_extract_defaults()` が `model` を抽出できない）。
  - `service.build_config()` が `model` キー存在時に `name` を補完しないため、`{"model": {"params": ...}}` がそのままバリデーションへ到達する。
  - UI は `Model Type` を `value.name ?? "lgbm"` で表示しており、実際の config 欠落を隠してしまう。
  - `adapter.validate_config()` がエラー詳細を落として返すため、UI 上で根因（`union_tag_not_found`）が見えにくい。
  - 既存テストは簡略 schema モック中心のため、`oneOf/const/discriminator` の実系統を検出できなかった。
- **影響範囲**:
  - Fit/Tune のデフォルト実行導線
  - Model タブ表示の信頼性
  - バリデーション失敗時のデバッグ容易性
  - 回帰テストの検知能力
- **対応先**:
  - PLAN.md Phase 11 に追補（11-7 / 11-8 / 11-9）

---

### P-007: `evaluation.params` フィールド追加（Widget-only, precision_at_k の k 値指定）

- **日付**: 2026-03-14
- **ステータス**: Approved（2026-03-14 承認）
- **背景**:
  - LizyML の `PrecisionAtK` メトリックは `k` パラメータ（デフォルト 10、範囲 1-100）を持つが、`EvaluationConfig` の schema は `metrics: list[str]` のみで `additionalProperties: false` のためパラメータ指定手段がない。
  - LizyML の `get_metric()` は常に `cls()` で引数なしインスタンス化するため、ライブラリ側でのカスタマイズ不可。
  - Widget ユーザーが k 値を指定できるようにするため、Widget-only の `evaluation.params` フィールドを追加する。
- **提案内容**:
  - Widget config の `evaluation` セクションに `params: dict` フィールドを追加（Widget-only、`strip_for_backend` で除去）。
  - 初期対応: `params.precision_at_k_k: int`（デフォルト 10、範囲 1-100）。
  - UI: Evaluation セクションで `precision_at_k` が選択されている場合のみ k 入力フィールドを表示。
  - Score 表示: metric 名が `precision_at_k` の場合、表示名に `(k=N)` を併記。
  - Score 表示: k 値は `precision_at_k (k=N)` として併記（表示のみ）。
  - **制約**: LizyML の `get_metric()` は常に `cls()` で引数なしインスタンス化するため、現時点では常に k=10 で評価される。LizyML 側が custom k パラメータをサポートした際に adapter で k 値を forward する予定。
- **影響範囲**:
  - `evaluation` config フィールドの追加（Widget-only）
  - `strip_for_backend` の更新
  - ConfigTab Evaluation UI の更新
  - ScoreTable の表示更新
- **BLUEPRINT 更新**: §5.3 Evaluation セクションに `params` フィールドを追記

---

### P-013: `classify_best_params` を `BackendAdapter` Protocol に追加

- **日付**: 2026-03-14
- **ステータス**: Proposed
- **背景**:
  - Tune 完了後の Apply to Fit で `best_params` を `model / smart / training` カテゴリに分類する `classify_best_params` メソッドが `LizyMLAdapter` に実装済み。
  - 現状 `WidgetService` は `getattr` による duck typing で呼び出しており、`BackendAdapter` Protocol に含まれていない。
  - 新規 Adapter 実装時にカテゴリ分類の契約が不明瞭になるリスクがある。
- **提案内容**:
  - `BackendAdapter` Protocol に `classify_best_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]` を追加。
  - 戻り値は `(model_params, smart_params, training_params)` の 3-tuple。
  - デフォルト実装を持たない Adapter は `(params, {}, {})` を返す（全パラメータを model カテゴリに分類）。
  - `WidgetService` の `getattr` フォールバックを通常のメソッド呼び出しに変更。
- **影響範囲**:
  - `BackendAdapter` Protocol の変更（`adapter.py`）
  - `WidgetService.classify_best_params` の簡素化（`service.py`）
  - 将来の Adapter 実装者への契約明示
