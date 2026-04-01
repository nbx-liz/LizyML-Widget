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
- **ステータス**: Approved（2026-03-17 承認）
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

---

### P-014: Fit/Tune タブ再設計（Fit 欠落項目修正 + Tune 独立設定 + 対応関係の明示）

- **日付**: 2026-03-15
- **ステータス**: Approved（2026-03-15 承認）
- **関連**: BLUEPRINT.md §5.3, §3.3, §3.4, §6.3
- **背景**:
  - Fit タブに LizyML `LGBMConfig` の 4 つの nullable フィールド（`balanced`, `feature_weights`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio`）が存在するが、DynForm の `anyOf` 解決が非 null バリアントを展開するだけで **null トグルを提供しない**ため、ユーザーが null（自動判定 / 無効）に設定できない。
  - `model.params` は `additionalProperties: true` だが、`TypedParamsEditor` に定義済みの項目（`parameter_hints` の 12 項目）以外を追加する UI がなく、LightGBM が受け付ける多数のパラメータを設定できない。
  - Tune タブは `tuning.optuna` 固有の設定（`n_trials`, `metric`, `space`）のみを表示し、`model.params`・`training`・`evaluation` は Fit タブの設定に**暗黙的に依存**している。ユーザーからは Tune がどの設定で実行されるか見えない。
  - LizyML 仕様上、Tune は `calibration.*` を**非参照**（Fit 専用）。一方、Smart Params（`model.auto_num_leaves`, `model.num_leaves_ratio`, `model.min_data_in_leaf_ratio`, `model.min_data_in_bin_ratio`, `model.feature_weights`, `model.balanced`）は Tune でも使用される（`resolve_smart_params` は毎 trial で呼ばれ、Search Space に `category="smart"` 次元を含められる）。※初期実装では Smart Params も Tune 非参照と誤認し除去していたが、Bug 7 で修正済み。
  - Tune Settings の `metric`（`tuning.optuna.params.metric`）は実質的に `evaluation.metrics` の先頭要素を選ぶ操作であり、Adapter 内で `MODEL_METRIC_TO_EVAL` 変換 + `evaluation.metrics` 並べ替えを行っているだけで独立した概念ではない。
- **提案内容**:

  **A. Fit タブ欠落修正**

  1. nullable フィールドを以下の UI で操作可能にする:
     - `min_data_in_leaf_ratio` / `min_data_in_bin_ratio`: `lzw-stepper` を常に表示する。値は常に数値。
     - `feature_weights`: `lzw-toggle` で ON/OFF。OFF = `null`。ON 時は列名セレクト（`df_info.columns` から `<select>` で選択）+ `lzw-stepper`（重み値）のペアを複数行追加できる構造化入力。
     - `balanced`: `lzw-toggle` で ON/OFF。OFF = `null`（自動判定）、ON = `true`。
  2. `TypedParamsEditor` の末尾に **Additional Params** セクションを配置する。パラメーター名は `<select>`（`backend_contract.ui_schema.additional_params` から候補を供給）、値は `lzw-stepper` で入力する。各行に `×` 削除ボタン。[+ Add] で行追加。`TypedParamsEditor` で描画済みのキーは選択肢から除外する。

  **Fit タブ UI（P-014 改訂後）:**

  ```
  ┌──────────────────────────────────────────────────┐
  │ [▶ Fit]  [ Tune]                  [━━ Fit ━━]   │  ← sticky
  ├──────────────────────────────────────────────────┤
  │ ▸ Model ─────────────────────────────────────── │
  │   Model Type     lgbm （読み取り専用）            │
  │                                                  │
  │   ── Smart Params（Fit 専用）──                   │
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
  │   Objective    [binary           ▼]  ← select    │
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
  │     Inner Validation [Default   ▼]               │
  │                                                  │
  │ ▸ Evaluation ────────────────────────────────── │
  │   metrics  [auc][logloss][f1][accuracy][...]    │  ← chip (multi)
  │                                                  │
  │ ▸ Calibration [●──] ────────────────────────── │  ← binary のみ。トグル左寄せ
  │   (ON 時:)                                       │
  │     method   [platt             ▼]               │
  │     n_splits [5               ]                  │
  │                                                  │
  │ [Import YAML]  [Export YAML]  [Raw Config]       │
  └──────────────────────────────────────────────────┘
  ```

  Smart Params の各フィールド挙動:
  - `auto_num_leaves`: 既存どおり。ON = ratio モード、OFF = num_leaves 直接指定。
  - `min_data_in_leaf_ratio` / `min_data_in_bin_ratio`: `lzw-stepper` を常に表示。初期値 `0.01`。
  - `feature_weights`: `lzw-toggle` で ON/OFF。OFF = `null`（無効）。ON 時は列名（`df_info.columns` から `<select>` で選択）+ 重み（`lzw-stepper`、初期値 `1.0`）のペアを複数行追加できる。各行に `×` 削除ボタン。[+ Add] で行追加。
  - `balanced`: `lzw-toggle` で ON/OFF。OFF = `null`（task に応じて自動判定）、ON = `true`（強制バランシング）。

  Additional Params の挙動:
  - `TypedParamsEditor` が描画するキー（`parameter_hints` 定義済み + `verbose`）を除外した `model.params` の残りを表示する。
  - パラメーター名は `<select>` で選択する。候補は `backend_contract.ui_schema.additional_params` から供給される（LightGBM が受け付けるパラメーター名のうち、`parameter_hints` に含まれないもの）。
  - 値は `lzw-stepper` で入力する。
  - 各行に `×` 削除ボタン。[+ Add] で行追加。

  **B. Tune タブ独立設定化**

  1. `config` traitlet の `tuning` セクションを拡張し、Tune 専用の共通設定を格納する:
     - `tuning.model_params: dict` — Search Space の Fixed model param 値を格納する。
     - `tuning.training: dict` — Search Space の Fixed training 値を格納する。
     - `tuning.evaluation: dict` — Tune 用 evaluation 設定（`metrics` 配列。先頭 = 最適化対象）
     - これらは Widget-only フィールドであり、`strip_for_backend()` で LizyML への送信前に除去される。
  2. Tune タブに以下の **3 セクション**を表示する:
     - **Tuning Settings** — `n_trials`
     - **Search Space** — Model Params と Training Params のベースライン値 + 探索空間を**統合管理**する。Fixed 行の Config 列がベースライン値、Range/Choice 行が探索空間。[+ Add] で任意パラメータを行追加可能。
     - **Evaluation** — Optimization Metric（セグメントボタン、単一選択）+ Additional Metrics（チップ、複数選択、任意）
  3. Tune タブから以下を**除外**する:
     - 独立した Model Params セクション（Search Space に統合）
     - 独立した Training セクション（Search Space に統合）
     - Calibration（Tune 非参照。Smart Params は Search Space で探索可能なため Fit タブ UI に表示）
     - Tune Settings の standalone `metric`（Evaluation セクションに統合）
  4. `tuning.optuna.params.metric` フィールドを**廃止**する。最適化対象メトリックは `tuning.evaluation.metrics[0]` から決定する。Tune の Evaluation が LizyML registry metric 名を直接使用するため、Adapter の `MODEL_METRIC_TO_EVAL` 変換は Tune 経路では不要になる。

  **Tune タブ UI（P-014 改訂後）:**

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
  │   │ objective        │ [Fixed|Choice] │ [binary       ▼] │  ← Fixed: select
  │   │ metric           │ [Fixed|Choice] │ [auc][bin_l][..] │  ← Fixed: chip(multi)
  │   │ n_estimators     │ [Fixed|Range ] │ [ - 1500 +     ] │  ← Fixed: stepper
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
  │   │ auto_num_leaves  │ [Fixed|Choice] │ [●──]            │
  │   │ num_leaves_ratio │ [Fixed|Range ] │ [ - 1.0 +      ] │
  │   │ min_data_in_l... │ [Fixed|Range ] │ [ - 0.01 +     ] │
  │   │ min_data_in_b... │ [Fixed|Range ] │ [ - 0.01 +     ] │
  │   │ balanced         │ [Fixed|Choice] │ [──●]            │
  │   │                  │                │                  │
  │   │ ── Training ─────┼────────────────┼──────────────────┤
  │   │ seed             │ Fixed          │ [ - 42 +       ] │
  │   │ early_stop.enable│ Fixed          │ [●──]            │
  │   │ early_stop.rounds│ [Fixed|Range ] │ [ - 150 +      ] │
  │   │ validation_ratio │ [Fixed|Range ] │ [ - 0.1 +      ] │
  │   │ inner_valid      │ Fixed          │ [Default     ▼]  │
  │   │                  │                │                  │
  │   │ (Range に切り替えた場合:)                             │
  │   │ n_estimators     │ [Fixed|▶Range] │[-600+] ~ [-2500+]│
  │   │                  │                │                  │
  │   │ (Choice に切り替えた場合:)                            │
  │   │ objective        │[Fixed|▶Choice] │ ☑bin ☑cross      │
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

  Tune タブ各セクションの挙動:

  **Tuning Settings:**
  - `n_trials`: `lzw-stepper`（min=1）。Optuna の試行回数。

  **Search Space（Model Params + Training 統合）:**
  - Search Space は Tune における **model params と training params のベースライン値 + 探索空間の統合管理場所**となる。
  - `backend_contract.ui_schema.search_space_catalog` から pre-populate された行を **Model Params** グループと **Training** グループのサブ見出しで視覚的に区切る。
  - [+ Add] で任意パラメータ（`backend_contract.ui_schema.additional_params` から `<select>` で選択）を行追加できる。
  - 各行の **Mode** 列によって Config 列の UI が変わる:
    - **Fixed**: パラメータの型に応じた入力コントロール（stepper / select / toggle / chip）で値を直接編集する。model params の Fixed 値は `tuning.model_params` に、training の Fixed 値は `tuning.training` に格納する。
    - **Range**: `low` / `high` の `lzw-stepper` ペアで探索範囲を指定する。`tuning.optuna.space` に格納される。
    - **Choice**: チップボタン（複数選択）で候補値を指定する。`tuning.optuna.space` に格納される。
  - Training 行の Mode 制約:
    - `seed` / `early_stopping.enabled` / `inner_valid`: **Fixed のみ**（Mode セグメント非表示）。
    - `early_stopping.rounds` / `validation_ratio`: Fixed / Range を選択可能。
  - Fixed 値の初期値は `initialize_config` 時に Fit の現在値（`model.params` / `training`）からコピーする。以降は Fit と独立。
  - Search Space に含まれないパラメータは Tune 実行時に backend default を使用する（Fit の値にはフォールバックしない）。

  **Evaluation:**
  - **Optimization Metric**: セグメントボタン（単一選択）。`tuning.evaluation.metrics[0]` に格納する。候補は `backend_contract.ui_schema.option_sets.metric[task]`（LizyML registry metric 名）。初期値は task 別の先頭 metric。`direction`（maximize / minimize）は選択されたメトリックに応じて Adapter が自動決定する。
  - **Additional Metrics**: チップボタン（複数選択、任意）。`tuning.evaluation.metrics[1..]` に格納する。候補は Optimization Metric と同じ option set から、選択済みの Optimization Metric を除いたもの。空でもよい。LizyML は全メトリックを計算するが、Optuna objective には使用しない。

  Fit / Tune タブ対応関係:

  | セクション | Fit タブ | Tune タブ | 備考 |
  |-----------|---------|----------|------|
  | Smart Params | `model.auto_num_leaves` 等 | Search Space 内 Smart Params グループ | Fit は専用フォーム、Tune は Search Space 内で探索。config は保持される |
  | Model Params | `model.params`（TypedParamsEditor + Additional Params） | Search Space 内 Model Params グループ | Fit は専用フォーム、Tune は Search Space 内で Fixed/Range/Choice 管理 |
  | Training | `training`（専用セクション） | Search Space 内 Training グループ | Fit は専用フォーム、Tune は Search Space 内で Fixed/Range 管理 |
  | Evaluation | `evaluation`（chip multi） | `tuning.evaluation`（segment + chip） | Tune は Optimization Metric を明示的に分離 |
  | Calibration | `calibration` | ─ | Fit 専用。Tune 非参照 |
  | Tuning Settings | ─ | `tuning.optuna.params` | Tune 専用 |

  **C. 対応関係の明示**

  1. Fit の Model Params / Training の各項目と、Tune の Search Space 内の**同名パラメータ行**が対応する。Search Space 内の Model Params / Training サブグループ見出しが、Fit タブのセクション構成と視覚的に対応する。
  2. Fit 専用セクション（Smart Params / Calibration）は Fit タブのみ、Tune 専用セクション（Tuning Settings / Search Space / Evaluation）は Tune タブのみに表示する。
  3. Search Space の Fixed 値と `tuning.evaluation` の**初期値は Fit の現在値からコピー**する（`initialize_config` 時）。以降は独立して編集可能。

  **D. Adapter `prepare_run_config(job_type="tune")` の変更**

  1. `tuning.model_params`（Search Space の Fixed model param 値）を `model.params` に置換する（Fit の `model.params` は参照しない）
  2. `tuning.training`（Search Space の Fixed training 値）を `training` に置換する
  3. `tuning.evaluation` を `evaluation` に置換する
  4. calibration を除去する（Smart params は LizyML が Tune でも使用するため保持。Bug 7 で修正）
  5. `evaluation.metrics[0]` を最適化対象メトリックとし、`direction` を自動設定する
  6. `tuning.model_params` / `tuning.training` / `tuning.evaluation` が未設定の既存 config では、Fit 側の値にフォールバックする（後方互換）

  **E. Backend Contract 拡張**

  - `ui_schema.additional_params`: LightGBM が受け付けるパラメーター名のうち `parameter_hints` および `search_space_catalog` に含まれないものを候補として供給する。Fit タブの Additional Params セクションおよび Tune タブの Search Space [+ Add] の `<select>` が使用する。
  - `ui_schema.search_space_catalog` に Training パラメータ行を追加する（`seed`, `early_stopping.enabled`, `early_stopping.rounds`, `validation_ratio`, `inner_valid`）。各行に `modes` と `group` 属性を持たせ、Training グループの表示とMode 制約（Fixed のみ / Fixed+Range）を制御する。

- **影響範囲**:
  - `config` traitlet 構造変更（`tuning` セクションに `model_params` / `training` / `evaluation` を追加）
  - `BackendAdapter.prepare_run_config()` の tune 処理変更
  - `BackendAdapter.initialize_config()` の tuning デフォルト生成
  - `adapter_schema.strip_for_backend()` に Widget-only `tuning` フィールドの除去を追加
  - DynForm の nullable 型サポート追加（null トグル UI）
  - ConfigTab.tsx の Fit / Tune 両サブタブの UI 再構成
  - `tuning.optuna.params.metric` の廃止
  - Apply to Fit（P-005）のスナップショット復元ロジック更新
- **互換性**:
  - Python API（`fit()`, `tune()`, `set_config()`, `load_config()`）の既存インターフェースを維持する。
  - `tuning.optuna.params.metric` を含む既存 config の import 時は、Adapter が `tuning.evaluation.metrics` に変換する legacy 互換を提供する。
  - `tuning.model_params` / `tuning.training` / `tuning.evaluation` が未設定の既存 config は、`prepare_run_config()` で Fit 側の値にフォールバックする。
- **代替案**:
  - Fit / Tune で `config` traitlet を完全に 2 つに分離する案。LizyML の単一 config 構造と乖離が大きくなるため不採用。
  - 共通設定を別タブ（Common タブ）に移す案。タブ数が増え操作コストが上がるため不採用。
  - Tune タブに Fit の値を読み取り表示のみ行う案。Tune 独立設定の要件を満たさないため不採用。
- **受け入れ条件**:
  - Fit タブで `balanced`, `feature_weights`, `min_data_in_leaf_ratio`, `min_data_in_bin_ratio` が null ↔ 具体値で切り替え可能。
  - Fit タブで `model.params` に任意パラメータを追加・削除可能。
  - Tune タブに Model Params / Training / Evaluation セクションが表示され、Fit と独立して編集可能。
  - Tune 実行が Tune タブ + Data タブの設定のみで完結し、Fit タブの設定に依存しない。
  - Tune の最適化対象メトリックが `tuning.evaluation.metrics[0]` から決定される。
  - Apply to Fit（P-005）が引き続き正しく動作する。
  - 既存テスト 394 件が回帰なくパスする。

---

### P-015: `plot_inference` を `BackendAdapter` Protocol に追加

- **日付**: 2026-03-17
- **ステータス**: Approved（2026-03-17 承認）
- **背景**:
  - 推論結果のプロット生成（`prediction-distribution`, `shap-summary`）を行う `plot_inference` メソッドが `LizyMLAdapter` に実装済み。
  - 現状 `WidgetService.get_inference_plot()` は `getattr(self._adapter, "plot_inference", None)` による duck typing で呼び出しており、`BackendAdapter` Protocol に含まれていない。
  - 新規 Adapter 実装時に推論プロット生成の契約が不明瞭になるリスクがある。
- **提案内容**:
  - `BackendAdapter` Protocol に `plot_inference(predictions: pd.DataFrame, plot_type: str) -> PlotData` を追加。
  - `WidgetService.get_inference_plot()` の `getattr` フォールバックを通常のメソッド呼び出しに変更。
- **影響範囲**:
  - `BackendAdapter` Protocol の変更（`adapter.py`）
  - `WidgetService.get_inference_plot` の簡素化（`service.py`）
  - 将来の Adapter 実装者への契約明示

---

### P-016: `cv_strategies` を `BackendContract` capabilities に追加

- **日付**: 2026-03-17
- **ステータス**: Approved（2026-03-17 承認）
- **背景**:
  - Widget の `_handle_update_cv` に CV strategy の有効値リストが `_VALID_STRATEGIES` として frozenset でハードコードされている。
  - Widget はバックエンド固有の知識を持つべきではなく、有効な strategy 一覧は Backend Contract から取得すべきである。
- **提案内容**:
  - `BackendContract.capabilities` に `cv_strategies` リストを追加（例: `["kfold", "stratified_kfold", "time_series", "group_time_series", "purged_time_series", "group_kfold"]`）。
  - Widget の `_handle_update_cv` は `self.backend_contract["capabilities"]["cv_strategies"]` から有効値を取得し、フォールバックとして現行のハードコード値を使用する。
  - Widget の `_VALID_STRATEGIES` クラス属性を削除する。
- **影響範囲**:
  - `adapter_contract.py` の `build_capabilities()` に `cv_strategies` 追加
  - `widget.py` の `_handle_update_cv` の strategy 検証ロジック変更
  - `BackendContract` の capabilities 構造変更

---

### P-017: LizyML v0.2.0 対応（TuneProgressCallback 統合 + stratified_group_kfold + calibration n_splits 非推奨）

- **日付**: 2026-03-18
- **ステータス**: Approved（2026-03-18 承認）
- **背景**:
  - LizyML v0.2.0 が `TuneProgressCallback` を導入し、Tune 中の trial ごとの進捗情報（`current_trial`, `total_trials`, `best_score`, `latest_score`）をコールバックで提供する。
  - 現状 Widget の Tune 進捗は `_run_with_cancel_polling` による 0.5 秒間隔のポーリングで、進捗メッセージが不正確（常に `"Processing..."`）。
  - LizyML v0.2.0 が新 split method `stratified_group_kfold` を追加。
  - LizyML v0.2.0 が `calibration.n_splits` を非推奨化（outer CV splits を再利用）。
  - LizyML v0.2.0 が `default_space()` の import パスを変更。
  - LizyML v0.2.0 の `TuningResult` が `best_model_params` / `best_smart_params` / `best_training_params` に 3分割（`best_params` property で backward compat 維持）。
- **提案内容**:
  - Adapter `tune()` 内で `TuneProgressCallback` を作成し `model.tune(progress_callback=...)` に渡す。Widget の `on_progress` コールバックへのブリッジとして機能。
  - `capabilities.cv_strategies` に `stratified_group_kfold` を追加。
  - `build_config` の split フィールド条件に `stratified_group_kfold` を追加。
  - `default_space()` の import パスを `lizyml.estimators.lgbm.defaults` に変更（旧パスを fallback）。
  - `calibration.n_splits` の UI に非推奨表示を追加。
  - `oof_coverage` を `FitSummary.metrics` dict 内に pass through（型変更なし）。
- **影響範囲**:
  - `adapter.py`: `tune()` 内の progress callback ブリッジ実装
  - `adapter_schema.py`: `default_space()` import パス変更
  - `adapter_contract.py`: `cv_strategies` に `stratified_group_kfold` 追加
  - `service.py`: `build_config` の split フィールド条件追加
  - `FitSubTab.tsx`: calibration `n_splits` の非推奨表示
  - `pyproject.toml`: `lizyml>=0.2.0` バージョンピン

---

### P-018: Google Colab 互換のジョブ進捗ポーリング機構追加

- **日付**: 2026-03-19
- **ステータス**: Approved（2026-03-19 承認）
- **背景**:
  - Google Colab 上で Widget の Fit/Tune ボタンをクリックしても、UI が更新されず処理が始まったように見えない。
  - 診断の結果、Colab の comm 実装がバックグラウンドスレッドからの Python→JS 通信（traitlet 書き込み・`self.send()`）を一切伝播しないことを確認した。
  - 現行の `_job_worker` はバックグラウンドスレッドから `status` / `progress` / `elapsed_sec` / `fit_summary` 等の traitlet を直接書き込んでおり、JupyterLab / VS Code では動作するが Colab では JS 側に反映されない。
  - BLUEPRINT §2 原則 4「環境非依存（Jupyter / Colab / VS Code のいずれでも動作する）」に違反している。
- **検証済みアプローチ**:
  - ❌ `call_soon_threadsafe` — Colab の ipykernel では無効
  - ❌ `self.send()` from BG thread — IOPub 経由でも BG スレッドからは不可
  - ✅ JS ポーリング（`model.send()` 双方向 msg:custom + JS 側 elapsed 補間 + CSS transition）
- **提案内容**:
  - **Python 側**:
    - `widget.py` の `__init__` に `self.on_msg(self._handle_custom_msg)` を追加
    - `_handle_custom_msg` で `poll` タイプのメッセージを受信し、現在の traitlet 値を `self.send()` で返す
    - 既存の BG スレッドからの traitlet 書き込みは変更しない（JupyterLab / VS Code 互換維持）
  - **JS 側**:
    - `useJobPolling(model)` フックを新規作成
    - `status === "running"` 時に 1000ms 間隔で `model.send({type: "poll"})` を送信
    - `msg:custom` の `job_state` 応答でローカル state を更新
    - JS 側 100ms タイマーで `elapsed_sec` を補間（滑らかな表示）
    - `status` が `completed` / `failed` になったらポーリング停止
    - JupyterLab では traitlet `change:` イベントが先に到達し、ポーリングは確認程度に動作
  - **App.tsx**:
    - polled state と traitlet 値をマージし、polled 値を優先して子コンポーネントに渡す
  - **ProgressView.tsx**:
    - プログレスバーに `transition: width 0.8s ease-out` を追加（CSS アニメーション）
- **影響範囲**:
  - `src/lizyml_widget/widget.py` — `on_msg` ハンドラ追加
  - `js/src/hooks/useJobPolling.ts` — 新規作成
  - `js/src/App.tsx` — polled state マージ
  - `js/src/components/ProgressView.tsx` — CSS transition 追加
- **変更しないもの**:
  - `_job_worker` / `_run_job` — BG スレッドの traitlet 書き込みはそのまま
  - Service / Adapter 層
  - Python API（`w.fit()` / `w.tune()`）
  - 個別 UI コンポーネントの props インターフェース

---

### P-019: ダークモード対応

- **日付**: 2026-03-19
- **ステータス**: Approved（2026-03-19 承認）
- **背景**:
  - 現行の CSS は `var(--jp-*, fallback)` パターンでフォールバック値にライトモード色をハードコードしている（約 100 箇所）。
  - JupyterLab のダークテーマでは `--jp-*` CSS 変数が切り替わるため部分的に対応できるが、フォールバック色がライト固定のためコンポーネントによっては白背景のまま残る。
  - Google Colab では `--jp-*` CSS 変数自体が提供されないため、常にフォールバック値（ライトモード色）が使われる。Colab のダークモードでは Widget が浮いて見える。
  - BLUEPRINT §2 原則 4「環境非依存」の一環として、ダークモードを適切にサポートする必要がある。
- **提案内容**:
  - **Phase 1: CSS 変数層の導入**
    - `.lzw-root` に Widget 固有の CSS 変数（`--lzw-bg`, `--lzw-fg`, `--lzw-border`, `--lzw-accent` 等）を定義
    - ハードコードされた約 100 箇所のカラー値を Widget 固有変数に置換
    - ライトモードでの変数値は現行と同一（見た目の変化なし）
  - **Phase 2: ダークモード対応**
    - `@media (prefers-color-scheme: dark)` でダークモード変数値を定義
    - JupyterLab の `--jp-*` 変数が存在する場合はそちらを優先（`var(--jp-layout-color0, var(--lzw-bg))`）
    - Colab / `--jp-*` 未提供環境では Widget 固有変数にフォールバック
  - **Phase 3: Plotly プロットのテーマ追従**
    - Plotly の `layout.template` をダークモード時に `plotly_dark` に切り替え
    - JS 側で `prefers-color-scheme` を検出し、プロット描画時に適用
- **影響範囲**:
  - `js/src/widget.css` — CSS 変数定義 + ダークモードメディアクエリ追加 + ハードコード色置換
  - `js/src/hooks/usePlot.ts` または Plotly レンダリング部 — テンプレート切替
  - フロントエンドの外部依存ライブラリの追加・削除はなし
- **変更しないもの**:
  - Python 側のコード
  - traitlets / Action 契約
  - コンポーネント構造・ロジック

### P-020: libgomp OpenMP プール親和性問題の回避（subprocess 実行戦略）

- **日付**: 2026-03-19
- **ステータス**: Proposed
- **背景**:
  - Linux 環境（WSL2 含む）で libgomp（GCC OpenMP）を使用する場合、OpenMP スレッドプールは最初に並列リージョンを実行したスレッドに束縛される（GCC bug #108494）。
  - Widget では main thread が lightgbm import / Dataset 作成時に先に OpenMP を使用するため、worker thread からの Fit/Tune 実行時に 50x の速度劣化が発生する。
  - `daemon=False`、`omp_set_num_threads()`、thread 内 warm-up 等は効果なし。同一プロセス内では回避不可能。
  - LLVM libomp（macOS デフォルト）および MSVC vcomp（Windows）にはこの制約がない。
- **検証済みアプローチ**:
  - ❌ `daemon=False` — プール親和性が原因のため効果なし（daemon=True と同じ 50x）
  - ❌ `omp_set_num_threads()` / `threadpoolctl` — ICV は設定されるがプール再利用に影響しない
  - ❌ Thread 内 warm-up / dummy 並列リージョン — 一度束縛されたプールは再割当て不可
  - ✅ `LD_PRELOAD=libomp` — libomp にはプール親和性バグがなく 2.3x（許容範囲）
  - ✅ `subprocess.Popen` — 新プロセスでは libgomp 状態が初期化され 1.5x（許容範囲）
- **提案内容**:
  - **Phase 1: 環境検知モジュール（`openmp_detect.py`）**
    - `is_libgomp_affected() -> bool` — Linux + libgomp を検知
    - `find_libomp_path() -> str | None` — libomp の共有ライブラリパスを探索
    - `get_execution_strategy() -> tuple[str, str | None]` — `("thread", None)` or `("subprocess", libomp_path)`
  - **Phase 2: Subprocess エントリポイント（`_subprocess_entry.py`）**
    - `python -m lizyml_widget._subprocess_entry` として実行
    - stdin: pickle of `{job_type, config, df_bytes}`
    - stdout: length-prefixed pickle messages（progress / result / error）
    - SIGTERM でキャンセル
  - **Phase 3: Subprocess ランナー（`subprocess_runner.py`）**
    - `run_job_subprocess()` — subprocess 起動、LD_PRELOAD 設定、データ転送、進捗読取、キャンセル管理
  - **Phase 4: Widget 統合**
    - `_run_job` で `self._execution_strategy` に基づき分岐
    - `_subprocess_job_worker` 新規追加
    - `WidgetService.get_dataframe()` / `load_model_from_path()` 追加
    - `LizyMLAdapter.load_model()` 実装
  - **Phase 5: テスト**
    - 検知、ランナー、Widget 統合の各レイヤーでユニット + 統合テスト
- **影響範囲**:
  - `widget.py` — `__init__`（検知キャッシュ）、`_run_job`（分岐追加）、`_subprocess_job_worker`（新規）
  - `service.py` — `get_dataframe()`、`load_model_from_path()` 追加
  - `adapter.py` — `load_model()` 実装
  - 新規ファイル: `openmp_detect.py`、`subprocess_runner.py`、`_subprocess_entry.py`
- **変更しないもの**:
  - traitlets / Action 契約（subprocess path も同じ traitlet を更新）
  - Colab ポーリング機構（`_handle_custom_msg` は traitlet 値を読むため変更不要）
  - フロントエンド（JS/CSS）
  - Windows / macOS の実行パス（`threading.Thread` のまま）

### P-021: CodeGen（コードエクスポート）機能

- **日付**: 2026-03-21
- **ステータス**: Accepted → Implemented（実装完了後にダウンロード方式を改善）
- **背景**:
  - LizyML v0.3.0 で `Model.export_code(path)` が追加された。LizyML に依存しない train.py / predict.py 等を生成できる。
  - Widget ユーザーが Fit で得た結果を本番パイプラインにそのまま持ち出せるようにする。
- **提案内容**:
  - **Adapter**: `BackendAdapter` Protocol に `export_code(model, path) -> Path` メソッド追加
  - **Service**: `WidgetService.export_code(path) -> Path` 追加
  - **Widget**: `_handle_export_code` アクションハンドラ追加。tmpdir に生成 → zip → `self.send(msg, buffers=[zip_bytes])` でバイナリバッファ送信
  - **JS**: Results タブに "Export Code" ボタン追加。`msg:custom` type `code_export_download` を受信し、`Blob` URL + `<a download>` クリックでブラウザの保存ダイアログを起動
  - **Python API**: `w.export_code(path=None) -> Path` 追加
- **実装時の変更点**（初期提案からの差分）:
  - `msg:custom` type を `code_export_result` → `code_export_download` に変更
  - ペイロードを `{path: string}` → `{filename: string}` + `buffers=[zip_bytes]`（バイナリバッファ転送）に変更
  - パス表示方式からブラウザダウンロードダイアログ方式に変更（JupyterLab / VS Code Notebook / Colab 全対応）
  - JS payload からパス指定を除去（セキュリティ対策）。Python API `w.export_code(path)` のみパス指定可
- **影響範囲**:
  - `adapter.py` — `BackendAdapter` Protocol 拡張、`LizyMLAdapter.export_code()` 実装
  - `service.py` — `export_code()` 追加
  - `widget.py` — `_handle_export_code` + `export_code()` 追加
  - `js/src/tabs/ResultsTab.tsx` — ボタン + msg:custom 応答（ブラウザダウンロード）
- **変更しないもの**:
  - traitlets（msg:custom のみ使用）
  - 既存の Fit / Tune / Predict フロー
  - CSS / ダークモード

### P-022: BlockedGroupKFold CV 戦略対応

- **日付**: 2026-03-21
- **ステータス**: Proposed
- **背景**:
  - LizyML v0.4.0 で `blocked_group_kfold` CV 戦略が追加された。時間軸（Period）× エンティティ軸（Group）の 2 軸交差検証。
  - 金融・広告・EC のパネルデータで「将来の時点 × 未知のユーザー」への汎化性能を正しく評価できる。
- **提案内容**:
  - **Contract**: `cv_strategies` リストに `blocked_group_kfold` 追加
  - **Service**:
    - `update_cv()` に blocks / groups ネストパラメータ追加
    - `build_config()` で nested split セクション生成
    - `get_column_stats(col)` 新規 — カラムの値分布を取得
    - `preview_splits()` 新規 — Fit 前の fold プレビュー計算
  - **Widget**:
    - `_handle_update_cv` バリデーション更新（blocks.col ≠ groups.col）
    - `_handle_get_column_stats` / `_handle_preview_splits` アクションハンドラ追加
  - **JS**:
    - `BlockedGroupKFold.tsx` 新規コンポーネント（Blocks + Groups + FoldPreview）
    - `DistributionBar.tsx` 新規（値分布棒グラフ）
    - `FoldPreview.tsx` 新規（fold 可視化）
    - DataTab に戦略ボタン追加 + 条件付き表示
    - `msg:custom` で `column_stats` / `split_preview` を受信
  - **df_info.cv 構造変更**:
    ```python
    "cv": {
        "strategy": "blocked_group_kfold",
        "blocks": {"col": "date", "cutoffs": [...], "mode": "expanding", "train_window": null},
        "groups": {"col": "user_id", "n_splits": 3, "stratify": "auto", "shuffle": true},
        "min_train_rows": 10,
        "min_valid_rows": 5,
    }
    ```
- **影響範囲**:
  - `adapter_contract.py` — cv_strategies 追加
  - `service.py` — update_cv / build_config 拡張、get_column_stats / preview_splits 新規
  - `widget.py` — アクションハンドラ追加
  - `js/src/tabs/DataTab.tsx` — CV_STRATEGIES + 条件付き表示
  - `js/src/components/` — 3 つの新規コンポーネント
  - `js/src/widget.css` — 新 UI スタイル
- **変更しないもの**:
  - 既存 CV 戦略の動作
  - Adapter Protocol（CV は Service 管理）
  - traitlets 定義（df_info 内の cv 構造変更のみ）

---

### P-023: Action 通信を traitlet 同期から msg:custom に移行（Colab ipywidgets 7.x 互換）

- **日付**: 2026-03-25
- **ステータス**: Proposed
- **背景**:
  - Google Colab（ipywidgets 7.7.1）で Fit ボタンをクリックしても処理が開始されない障害が発生。
  - 診断の結果、JS→Python 方向の `Dict` traitlet 同期（`model.set("action", {...})` + `model.save_changes()`）が Python 側の `@traitlets.observe("action")` に到達しないことを確認。
  - Python 側から直接 `w.action = {...}` を設定すると Fit は正常に完了する。BG スレッドからの Python→JS traitlet 同期は動作する。
  - P-018 で導入したポーリング機構の `isColab()` 検出（`link[href*="colab"]`）も `false` を返すようになっているが、BG スレッド通信自体は現行 Colab で動作するため、ポーリングの必要性は低下。
  - anywidget issue [#786](https://github.com/manzt/anywidget/issues/786) で Dict traitlet の初回同期失敗が報告されており、ipywidgets 7.x 固有の挙動と推定。
- **提案内容**:
  - **JS 側**:
    - `useSendAction` hook を `model.set()` + `save_changes()` から `model.send()` (msg:custom) に変更
    - `usePlot.ts` の `requestPlot` も同様に `model.send()` に変更
    - `action` traitlet の `useTraitlet` 購読を削除（不要になるため）
  - **Python 側**:
    - `_handle_custom_msg` で `type: "action"` のメッセージを受信し、既存の action dispatch ロジック（`_action_handlers` map）に委譲
    - `@traitlets.observe("action")` の `_on_action` は Python API 互換のため維持（`w.action = {...}` による直接操作を引き続きサポート）
    - `action` traitlet 定義自体は残す（Python API の後方互換性維持）
  - **isColab() 検出の改善**:
    - `window.google?.colab` を primary check に追加、`link[href*="colab"]` を fallback に格下げ
  - **ポーリング機構の改善**:
    - BG スレッド通信が動作する環境ではポーリング不要だが、将来の Colab 変更に備えてフォールバックとして維持
    - stall detection（traitlet 更新が 2 秒間来ない場合にポーリング開始）を Colab 限定から全環境に適用（環境検出不要に）
- **影響範囲**:
  - `js/src/hooks/useModel.ts` — `useSendAction` の通信方式変更
  - `js/src/hooks/usePlot.ts` — `requestPlot` の通信方式変更
  - `js/src/hooks/useJobPolling.ts` — `isColab()` 検出改善 + stall detection 汎用化
  - `js/src/App.tsx` — `action` traitlet 購読削除（軽微）
  - `src/lizyml_widget/widget.py` — `_handle_custom_msg` にアクション dispatch 追加
- **変更しないもの**:
  - `action` traitlet 定義（Python API 後方互換性のため残す）
  - `_on_action` observer（Python API 経由の操作をサポート）
  - `_action_handlers` map（dispatch ロジックは共通）
  - Service / Adapter 層
  - 個別 UI コンポーネントの props（`sendAction` の型シグネチャは変わらない）
  - ポーリング応答ロジック（`_handle_custom_msg` の `poll` ハンドラ）
- **受け入れ基準**:
  - Colab（ipywidgets 7.7.1）で Fit / Tune / 全アクションが動作する
  - JupyterLab / VS Code Notebooks での動作に退行がない
  - Python API（`w.action = {...}`）が引き続き動作する
  - 既存テストが全パス + 新規テストで msg:custom action dispatch をカバー

---

### P-024: `load_model` / `model_info` Python API

- **日付**: 2026-03-27
- **ステータス**: Proposed
- **背景**:
  - ユーザーが過去に学習・保存したモデルを Widget に読み込み、推論やプロット取得を行いたいケースがある。
  - 現状 `save_model()` / `export_model()` はあるが、保存したモデルを再ロードして Widget に復元する Python API が存在しない。
  - `BackendAdapter` Protocol には `load_model(path)` / `model_info(model)` が定義済みだが、Widget 層のパブリックメソッドとして公開されていない。
  - `model_info` は `NotImplementedError` を送出する未実装状態。
- **提案内容**:
  - `LizyWidget.load_model(path: str) -> LizyWidget` — `_service.load_model_from_path(path)` を呼び出し、`status = "completed"` に設定、`available_plots` を更新する。
  - `LizyWidget.model_info: dict[str, Any] | None` プロパティ — モデルが存在すれば安全なメタデータ dict を返す。モデル未ロード時は `None`。
  - `LizyMLAdapter.model_info(model)` — `NotImplementedError` を `{"loaded": True}` + パラメータ情報の返却に変更。
- **影響範囲**:
  - `src/lizyml_widget/widget.py` — パブリックメソッド・プロパティの追加
  - `src/lizyml_widget/adapter.py` — `model_info` の実装
- **受け入れ基準**:
  - `load_model(path)` で `status == "completed"` かつ `available_plots` が取得される
  - `model_info` がモデル未ロード時に `None`、ロード後に `dict` を返す
  - 既存テスト全パス + 新規テストでカバー

---

### P-026: Learning Curve Plot に metrics フィルタを追加

- **日付**: 2026-03-28
- **ステータス**: 承認・実装
- **バグ修正（実装中に発見）**:
  - `model_metric` option set の binary に `logloss`, `auc_pr`, `f1`, `accuracy`, `brier` が含まれていたが、
    これらは LizyML 評価 metric 名であり LightGBM ネイティブ metric 名ではない。
    LightGBM がサイレントに無視するため Learning Curve に表示されなかった。
    LightGBM ネイティブ名のみに修正: `auc`, `binary_logloss`, `binary_error`, `average_precision`。
  - regression の `r2`, `rmsle` も LightGBM ネイティブではないため除去。
  - `MODEL_METRIC_TO_EVAL` に `binary_error` → `accuracy`, `average_precision` → `auc_pr` を追加。
  - PlotViewer の `onRequest` を `useCallback` で安定化し、metric フィルタが上書きされないよう修正。
- **背景**:
  LizyML v0.5.0 で `plot_learning_curve(metrics=...)` フィルタ引数が追加された（LizyML#52）。
  Widget 幅（~600px）では metric 3個以上の横並び subplot が溢れるため、
  Widget 側で metric 選択 UI を提供し、Python 側でフィルタして返す仕組みが必要。
- **提案内容**:
  - `BackendAdapter.plot()` のシグネチャに `**kwargs` を追加（learning-curve の `metrics` 等を透過）
  - `WidgetService.get_plot()` に `**kwargs` を追加して Adapter に透過
  - `LizyMLAdapter.plot()` で `learning-curve` の場合に `metrics` kwarg を `plot_learning_curve()` に渡す
  - `Widget._handle_request_plot()` が payload の `options` dict を `get_plot()` に透過
  - JS: `request_plot` action の payload に `options: { metrics?: string[] }` を追加
  - JS: ResultsTab で Learning Curve 選択時に metric セレクタ UI（セグメントボタン）を表示
  - デフォルトは先頭 metric のみ表示（Widget 幅制約）、全選択も可能
- **影響範囲**:
  - `BackendAdapter` Protocol: `plot()` に `**kwargs` 追加（後方互換）
  - `service.py`: `get_plot()` に `**kwargs` 透過
  - `adapter.py` (`LizyMLAdapter`): `plot()` で kwargs を learning-curve に渡す
  - `widget.py`: `_handle_request_plot` が payload.options を透過
  - `js/src/hooks/usePlot.ts`: `requestPlot` に options パラメータ追加
  - `js/src/tabs/ResultsTab.tsx`: Learning Curve 用 metric セレクタ追加
- **互換性**:
  - `**kwargs` のため既存の Adapter 実装は変更不要（後方互換）
  - `options` が省略された場合は従来通り全 metric 表示
  - traitlets 変更なし、msg:custom payload の拡張のみ
- **代替案**:
  - JS 側で Plotly subplot を表示切替 → サーバー側フィルタの方がデータ転送量が少なく実装がシンプル
  - 全 metric を常に送信 → Widget 幅超過の根本問題が解決しない
- **受け入れ基準**:
  - Learning Curve リクエスト時に metrics フィルタが LizyML に渡される
  - ResultsTab に Learning Curve 用 metric セレクタが表示される
  - デフォルトで先頭 metric のみ表示、切替可能
  - 既存テスト全パス + 新規テスト追加

---

### P-025: CV Strategy Metadata in Backend Contract + Service Default Delegation

- **日付**: 2026-03-27
- **ステータス**: 承認・実装
- **目的**:
  JS の DataTab に残る CV 戦略固有の定数 (NEEDS_GROUP, NEEDS_TIME 等) を backend_contract.capabilities に移動し、
  Service の CV デフォルトロジックを Adapter に委譲する。
- **影響範囲**:
  - `BackendContract.capabilities` に `cv_strategy_fields` / `cv_defaults` / `cv_default_strategy` を追加
  - `adapter_contract.py` の `build_capabilities()` を拡張
  - `service.py` の `_default_cv_state` / `_default_strategy_for_task` を adapter contract 経由に変更
  - `js/src/tabs/DataTab.tsx` — contract から CV strategy fields を読み取り、ハードコード値をフォールバックに格下げ
  - `js/src/components/SearchSpace.tsx` — `special_search_space_fields` を ui_schema から読み取り
- **互換性**:
  - JS: backend_contract から読み取り、フォールバックでハードコード値を保持
  - Python: build_capabilities に追加のみ（Adapter Protocol 変更なし）
- **代替案**:
  - 完全に JS 側のハードコードを維持 → backend 追加時にJS変更が必要になり拡張性が低い
- **受け入れ基準**:
  - DataTab が backend_contract.capabilities.cv_strategy_fields を使用
  - Service の CV デフォルトが adapter contract 経由
  - SearchSpace の special field 判定が ui_schema 経由
  - 既存テスト全パス
