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
