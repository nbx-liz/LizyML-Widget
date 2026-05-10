## LizyML-Widget 仕様変更履歴

### P-039: libgomp / OpenMP プール親和性回帰の体系的予防（policy(openmp,perf) — #147 / #154 / #156 retrospective）

- **日付**: 2026-05-10（提案・Phase 1 完了 / Phase 2 着手中）
- **ステータス**: 採択（Phase 1: PR #162 で develop マージ済 / Phase 2: 着手中 / Phase 3-4: 着手前）
- **関連 Issue**: [#160](https://github.com/nbx-liz/LizyML-Widget/issues/160)（本 Proposal の元 issue）, [#147](https://github.com/nbx-liz/LizyML-Widget/issues/147)（P-036 motivating case）, [#154](https://github.com/nbx-liz/LizyML-Widget/issues/154)（PR #155 motivating case）, [#156](https://github.com/nbx-liz/LizyML-Widget/issues/156)（P-038 motivating case）, [#158](https://github.com/nbx-liz/LizyML-Widget/issues/158)（同累積劣化、follow-up）
- **背景**:
  - libgomp プール親和性問題（GCC bug [#108494](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108494)）に起因する 10-50x スローダウンが、本 codebase で **4 回** 異なる近接症状として再発した。各回ごとに reactive な修正（P-020, P-036, PR #155 thread fallback, P-038 subprocess retune resume）を入れ、learned skill を書き、specific path を pin する regression test を追加してきたが、**新規 ML 呼出経路を追加することで同種クラスの回帰を再導入できる構造的弱点が残っている**。
  - 同一 root cause: GCC libgomp は OpenMP parallel region に最初に入った thread に thread pool を bind する。LightGBM / SHAP は OpenMP を多用する。Jupyter / anywidget 環境では `widget._run_job → ThreadJobRunner`, `widget.predict(df)`, `widget_actions.handle_run_inference → service.predict`, `adapter.plot(model, "feature-importance-shap")` のいずれかが parent main thread で OpenMP region に入った瞬間、後続の worker-thread 上の Tune / Fit / Retune が catastrophe path に入る。
  - PR #155 thread fallback / P-038 subprocess retune resume / P-036 subprocess default は **既知の call site** を 1 つずつ塞いだが、**将来追加される 5 つ目の call site** を構造的に防ぐ仕組みは無い。
- **提案内容**: 段階的な防御層（defense in depth）。各 Layer は独立に着手可能だが、Phase の順序で着手する。
  - **Layer 1（Architectural — Phase 3）**: `BackendExecutor` 抽象を `WidgetService` 配下に新設し、すべての ML library 呼出（predict / SHAP / model 内部 plot 等）を必ず通す。executor は execution strategy を見て inline / subprocess / dedicated-OpenMP-owner-thread を選ぶ。`service.predict` / `service.get_inference_plot`（SHAP 含む経路）/ `adapter.plot(... feature-importance-shap)` を funnel する。
  - **Layer 2（Runtime invariant guard — Phase 2）**: BLUEPRINT.md §6.4 に **INV-G** を追加 — "post-subprocess-tune において parent main thread の libgomp parallel region が次の worker-thread Tune/Fit/Retune より先に発生してはならない"。`service._libgomp_pool_owner_known: Literal["subprocess","main","worker","unknown"]` を追加し、`SubprocessJobRunner.run` 完了時に `"subprocess"` をセット、parent main thread の predict/SHAP 経路で `"main"` をセット、worker-thread Tune/Fit 起動前に検査して `"main"` の場合 subprocess に re-route または WARN log。
  - **Layer 3（Test grid — Phase 1）**: `tests/regression/test_reg_160_libgomp_perf_grid.py` を新設。`{intermediate ∈ noop / main_thread_predict / main_thread_fit_predict}` × `{next_op ∈ retune / fit}` の cross-product で per-trial wall-clock を baseline subprocess tune の 1.5x 以内に pin。#154（clean → retune）と #156（predict → retune catastrophe）の両方を grid の異なる cell で再現可能にする。
  - **Layer 4（CI — Phase 1）**: `.github/workflows/ci.yml` に `libgomp-perf` job を追加し `ubuntu-latest`（libgomp by default）で `pytest -m slow tests/regression/test_reg_160_libgomp_perf_grid.py` を毎 PR 実行。catastrophe を CI ブロッカーにする。既存 `test_reg_147_openmp_perf.py` / `test_reg_156_subprocess_retune.py` は dataset サイズが GitHub runner にとって過大なため別経路（developer / nightly）で動かす（本 Proposal では grid のみを CI gate に組み込む）。
  - **Layer 5（Change-gate — Phase 4）**: `~/.claude/rules/project/change-gate.md`（global rule）と `CLAUDE.md` に "ML library への直接呼出（lightgbm / shap / xgboost / sklearn）を `BackendExecutor` 外部に追加する" を gate-required category として追記。pre-commit / ruff custom rule で `import lightgbm` / `import shap` / `model.predict` 等が executor module 外部に出現したら fail。`# noqa: ML-CALL` allowlist comment で例外承認。
- **Phase 構成と本 Proposal のスコープ**:
  - **Phase 0**: 本 Proposal の HISTORY.md 追加（**本コミットで完了**）
  - **Phase 1**: Layer 3 + Layer 4（**本 PR で着手** — CI gate と parameterised grid）
  - **Phase 2**: Layer 2（INV-G + 実行時ガード — 別 PR）
  - **Phase 3**: Layer 1（`BackendExecutor` refactor — 別 PR、blast radius 最大）
  - **Phase 4**: Layer 5（change-gate codification + lint rule — 別 PR）
- **Invariants（本 Proposal が宣言する将来不変条件）**:
  - INV-G（Phase 2 で encode）: post-subprocess-tune において、parent main thread の libgomp parallel region が次の worker-thread Tune/Fit/Retune より先に観測されない（観測された場合は subprocess に re-route または WARN）。
  - INV-H（Phase 3 で encode）: lightgbm / shap / xgboost / sklearn API への直接呼出は `BackendExecutor` モジュール外部から発生しない（type system + lint で enforce）。
  - INV-I（Phase 1 で encode、本 PR）: `tests/regression/test_reg_160_libgomp_perf_grid.py` の各 cell について per-trial wall-clock が baseline subprocess tune の 1.5x 以内（catastrophe 検出のみ。absolute perf budget は別 grid で扱う）。
- **Phase 1 影響範囲（本 PR）**:
  - `tests/regression/test_reg_160_libgomp_perf_grid.py` — 新規（parameterised grid + 1.5x bound）
  - `.github/workflows/ci.yml` — 新規 job `libgomp-perf` を追加
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]` セクションに Phase 1 entry を追加
- **互換性**:
  - 公開 API・traitlets・JS UI の変更なし（Phase 1 は test + CI のみ）
  - `pytest -m slow` は引き続き opt-in。新 grid は CI で自動実行されるが、ローカル開発では `-m slow` を明示しない限り skip
- **代替案（却下）**:
  - **案A**: Phase 1 で既存 `test_reg_147` / `test_reg_156` をそのまま CI に乗せる → GitHub Actions 2-core runner で 100k × 50 dataset は per-job 30 分超になる。Phase 1 で別 grid を新設し dataset を 5k × 20 まで縮小する方が PR latency 影響が小さい。catastrophe は dataset サイズに依存しないので 1.5x bound は同じ表現力を持つ。
  - **案B**: Phase 3（`BackendExecutor`）から先に着手 → blast radius が公開 Python API（`w.predict(df)` の async 化検討）まで及ぶ。既知 call site の incremental 修正（P-036 / P-038）と異なり、Proposal 単位の合意形成と段階的 migration が必要。Phase 1 が CI を gate に入れる時点で historical regression class は捕捉できるため、Phase 3 を急ぐ便益が小さい。
  - **案C**: `LD_PRELOAD=libomp` の wrapper script 配布で GCC bug 自体を回避 → ユーザの環境管理に踏み込みすぎる（Out of scope per #160）。
- **受け入れ基準（Phase 1）**:
  - [ ] `tests/regression/test_reg_160_libgomp_perf_grid.py` が `intermediate × next_op` の cross-product を `@pytest.mark.parametrize` で表現する。
  - [ ] 各 cell の per-trial wall-clock が baseline subprocess tune の 1.5x 以内であることを assert する（libgomp 非搭載環境では skip）。
  - [ ] `.github/workflows/ci.yml` に `libgomp-perf` job が存在し、`pull_request` トリガで `ubuntu-latest` 上で grid を実行する。
  - [ ] grid を意図的に regress させると（例: 1.5x bound を 1.0x に絞る）CI が fail することを手元で確認する。
  - [ ] 既存品質ゲート（`pytest`, `ruff check`, `ruff format --check`, `mypy --strict`, `pnpm test:coverage`, `pnpm lint`）全 green。
- **本 Proposal の終了条件**:
  - Phase 1 〜 Phase 4 すべてが develop にマージされる。本 Proposal 自体は Phase 1 PR がマージされた時点で「decision: 採択」とし、Phase 2-4 はそれぞれ独立 PR で着手する。

### P-038: subprocess Retune Resume を P-037 の tune state IPC を input 方向に拡張して実装（#154 thread fallback の置換）

- **日付**: 2026-05-09（提案・決定・実装）
- **ステータス**: 決定（実装中 — fix/issue-154-default-retune → develop, PR #155 を Option A に転換）
- **関連 Issue**: [#156](https://github.com/nbx-liz/LizyML-Widget/issues/156)（PR #155 follow-up）, [#128](https://github.com/nbx-liz/LizyML-Widget/issues/128)（subprocess retune の本フィックス、本 Proposal で完了）, [#154](https://github.com/nbx-liz/LizyML-Widget/issues/154)（PR #155 の元 issue）
- **背景**:
  - PR #155（`e286626`）は `widget._run_job` で retune を ThreadJobRunner にフォールバックさせる **暫定 hotfix** を入れた。`w.tune() → w.retune()` の最低限の正常系は復帰したが、構造的な問題が残った。
  - [#156 の調査](https://github.com/nbx-liz/LizyML-Widget/issues/156#issuecomment-4412462760)で empirical に確認した内容:
    1. **Clean path のみで 1.43x 劣化**（subprocess tune 3.24 s/trial vs thread retune 4.64 s/trial）。issue#156 本文の 1.96x はこの band。
    2. **典型的ユーザフローで 11x 劣化（30x catastrophe の再発）**。`w.tune()` の後に `w.predict(test_df)` を 1 回呼ぶだけで、retune 中の親プロセス CPU が ~29 cores（2904%）から ~2.25 cores（225%）に崩壊し per-trial wall が 36 s に達する（S5/S7 で実測：109 s vs 14 s）。原因は GCC #108494 libgomp プール親和性バグ。`booster.predict` 単独でも親 main thread を pool owner に bind するのに十分（`learned/openmp-daemon-thread-degradation`）。
    3. **同一カーネル内累積劣化**（5.20 s → 9.04 s/trial across 4 cycles）。スレッド数は flat、CPU は full、原因不明。issue #158 で別途追跡。
  - PR #155 の thread fallback は構造上回避不可能なリスクを抱える。`w.predict` / Inference タブ / SHAP plot のいずれかが tune と retune の間に挟まれた瞬間に catastrophe path に入るが、ユーザがそれを避ける手段はない。
  - P-037（PR #153）で **subprocess→parent の tune state IPC**（`tune_state_out_path` + `BackendAdapter.export_tune_state` / `restore_tune_state`）を既に整備済み。本 Proposal はこれを **parent→subprocess の方向**にも拡張するだけで完結する。
- **提案内容**:
  - **(a) IPC 入力に `tune_state_in_path` と `retune_kwargs` を追加**:
    - `_subprocess_entry.read_input` が受け取る dict に 2 フィールドを追加:
      - `tune_state_in_path: str | None` — retune 時に親が事前に書き出した tune state の path（pickle blob, P-037 と同 format）
      - `retune_kwargs: dict[str, Any] | None` — `{"resume": True, "n_trials": int|None, "expand_boundary": bool|None, "boundary_threshold": float}`
    - `subprocess_runner.run_job_subprocess` の signature に同 2 引数を追加。
  - **(b) subprocess エントリの retune ブランチ**:
    - `retune_kwargs is not None` のとき、`adapter.create_model(config, df)` の直後に `adapter.restore_tune_state(model, tune_state_in_path)` を呼び、`adapter.tune(model, on_progress=on_progress, **retune_kwargs)` を実行する。
    - retune 後の更新済み tune state は既存の `tune_state_out_path` 経路で親に返す（P-037 と対称）。
    - subprocess 内の挙動は `service.tune(resume=True)` と論理同等（adapter.tune は既に resume kwargs を受け付ける）。
  - **(c) Service 拡張**: `WidgetService.export_tune_state_to_path(path)` を追加。`_tune_model` が None の場合は `ValueError` を上げ、subprocess retune を防ぐ defensive guard とする。
  - **(d) `SubprocessJobRunner.run` の retune 対応**:
    - `RetuneSubprocessUnsupportedError` の raise を **削除**。
    - `spec.retune_kwargs is not None` のとき:
      1. tempfile で `tune_state_in_path` を確保。
      2. `service.export_tune_state_to_path(tune_state_in_path)` を呼ぶ。
      3. `run_job_subprocess(..., retune_kwargs=spec.retune_kwargs, tune_state_in_path=tune_state_in_path)` で投げる。
      4. `finally` で `tune_state_in_path` を削除（既存の `tune_state_out_path` cleanup と対称）。
    - retune の場合も `record_subprocess_tune_summary` および `restore_tune_state_from_path` の post-subprocess 経路を流用する（既存実装で動作する）。
  - **(e) widget._run_job の thread fallback を撤去**:
    - PR #155 が追加した `_retune_fallback_warned` フラグおよび `if self._execution_strategy == "subprocess" and retune_kwargs is not None:` ブランチを削除。
    - runner 選択は execution strategy のみで決まる（retune_kwargs を見ない）。
  - **(f) 既存テストの更新**:
    - `tests/regression/test_reg_154_default_strategy_retune.py::INV-#154-A` の意味を **"retune は subprocess を使う"** に反転。
    - `INV-#154-B`（`LZW_FORCE_THREAD` 指示文言）は `RetuneSubprocessUnsupportedError` 全削除のため削除。
    - `INV-#154-C`（subprocess Fit overhead 30s bound）は無関係なので維持。
  - **(g) 新規 regression test**:
    - `tests/regression/test_reg_156_subprocess_retune.py`:
      - `INV-#156-A`: `w.tune() → w.retune()` がデフォルト install で `subprocess` runner 経由で完了する（`SubprocessJobRunner.run` 呼び出しを mock で検証）。
      - `INV-#156-B`（slow）: 100k × 50, 3 trials, `tune → main-thread booster.predict → retune` の per-trial wall-clock が clean subprocess tune の 1.5x 以内。catastrophe path を pin する（`learned/promote-learned-skill-to-regression-test` 適用）。
      - `INV-#156-C`（slow）: clean tune→retune の per-trial wall-clock が clean subprocess tune の 1.2x 以内。
- **Invariants**:
  - INV-1: subprocess retune を実行する条件下では、親プロセスは lightgbm を `import` のみで `train` / `predict` を呼ばない（subprocess が全並列領域を担う）。
  - INV-2: `service.export_tune_state_to_path` は `_tune_model is None` のとき必ず `ValueError` を上げる（subprocess retune の前提逸脱を early fail）。
  - INV-3: `tune_state_in_path` は `SubprocessJobRunner.run` の `finally` で必ず削除される（INV-5 の対称、ファイルリーク防止）。
  - INV-4: subprocess retune 完了後、親プロセスの `_tune_model._tuning_result.rounds` は元の rounds + retune で追加された rounds を含む（`restore_tune_state_from_path` 経由で復元）。
  - INV-5: `RetuneSubprocessUnsupportedError` は本 Proposal で削除する。BackendAdapter Protocol が `export_tune_state` / `restore_tune_state` を required としているため、retune を支えられない adapter は型レベルで存在しない。
- **影響範囲**:
  - `src/lizyml_widget/_subprocess_entry.py` — input dict に 2 フィールド、retune branch 追加
  - `src/lizyml_widget/subprocess_runner.py` — `run_job_subprocess` signature 拡張、input pickle に 2 フィールド追加
  - `src/lizyml_widget/job_runner.py` — `SubprocessJobRunner.run` で `RetuneSubprocessUnsupportedError` 削除、retune 経路追加；`RetuneSubprocessUnsupportedError` クラス自体は削除し import を整理
  - `src/lizyml_widget/service.py` — `export_tune_state_to_path` メソッド追加
  - `src/lizyml_widget/widget.py` — `_retune_fallback_warned` フラグと thread fallback ブランチを削除；`RetuneSubprocessUnsupportedError` の `except` 句は削除（type 削除に伴う）
  - `tests/regression/test_reg_154_default_strategy_retune.py` — INV-A 反転、INV-B 削除、INV-C 維持
  - `tests/regression/test_reg_156_subprocess_retune.py` — 新規（catastrophe path pin + clean perf bound）
  - `tests/test_subprocess_integration.py` — happy-path retune の mock テスト追加
  - `tests/test_widget.py` / `tests/test_widget_threading.py` — `RetuneSubprocessUnsupportedError` 参照の更新（残っていれば削除）
  - `BLUEPRINT.md` §3.7.1 — subprocess IPC ペイロードの input 方向追加を記載
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[0.9.0]` セクションの「Bug fixes」に追加（PR #155 のエントリを置換）
- **互換性**:
  - 公開 Python API（`w.fit/tune/retune`）の signature / 振る舞いは変更なし。`w.retune()` のユーザ可視動作は PR #155 の v0.9.0 直前状態と同一（成功する／status=`completed`）。
  - 副次的に `RetuneSubprocessUnsupportedError` クラスを削除する。`from lizyml_widget.job_runner import RetuneSubprocessUnsupportedError` は ImportError になるが、これは internal class（BLUEPRINT で公開 API として宣言されておらず、テストのみが参照していた）。CHANGELOG で internal change として明記。
  - subprocess IPC は input pickle に 2 フィールド追加。古い親 + 新しい subprocess の組み合わせは存在しない（subprocess は親と同じ Python 環境で起動する）ので skew リスクなし。
- **代替案（却下）**:
  - **案A**: PR #155 の thread fallback を維持し、`w.predict` の前後で警告を出す → ユーザ教育に頼るのは脆弱、誤って踏むと 11x 劣化、現在の v0.9.0 リリースを正当化できない。
  - **案B**: PR #155 を revert し `RETUNE_SUBPROCESS_UNSUPPORTED` を再露出 → #154 の元症状に戻る、最悪。
  - **案C**: lizyml 0.12 の resumable Optuna SQLite storage（issue #129）に依存 → P-030 でリリーススコープから外しており UI 露出が無い。本 Proposal で扱う in-memory tune state pickle で十分目的を達せるため不要。
- **Optuna study の扱い**:
  - P-037 と同じ。subprocess は受け取った `_study` を P-037 の `restore_tune_state` 経由で attach し、`adapter.tune(resume=True)` を呼ぶ。lizyml 内部で `_study` を使って resume する。
  - InMemoryStorage（既定）は pickleable。RDB-backed storage の場合 P-037 と同様に best-effort で `study=None` に降格し、retune は新規 study として動作する（functional には正しい動作）。
- **受け入れ基準**:
  - [ ] Linux + libgomp デフォルト環境で `w.tune() → w.retune()` が `subprocess` runner で完了する（mock で `SubprocessJobRunner.run` 呼び出しを検証）。
  - [ ] `tune → main-thread booster.predict → retune` の per-trial wall-clock が `tune` の per-trial wall の 1.5x 以内に収まる（issue #156 S7 の 11x catastrophe を除去）。slow regression test で pin。
  - [ ] clean `tune → retune` の per-trial wall-clock が `tune` の 1.2x 以内（PR #155 の 1.43x も改善）。
  - [ ] `RetuneSubprocessUnsupportedError` および `RETUNE_SUBPROCESS_UNSUPPORTED` エラーコードへの参照が widget / supervisor / tests から完全に消える。
  - [ ] PR #155 が追加した `_retune_fallback_warned` / "re-tune temporarily falls back to thread runner" ログが消える。
  - [ ] 既存品質ゲート: `pytest`, `pytest -m slow`, `ruff check`, `ruff format --check`, `mypy --strict`, `pnpm test:coverage`, `pnpm lint` 全 green。
  - [ ] CHANGELOG `[0.9.0]` の retune 関連エントリが PR #155 の "thread fallback" 記述から本 Proposal の "subprocess retune resume" 記述に置換される。
- **本 Proposal の終了条件**:
  - 受け入れ基準を満たした PR が `develop` にマージされる。
  - PR #155 の不採用部分（thread fallback コード）が同じ PR で回収される。

### P-037: Adapter Protocol に Tune State Cross-Process Persistence を追加（subprocess tune 後の plot 復元）

- **日付**: 2026-05-09（提案・決定・実装）
- **ステータス**: 決定（実装中 — fix/issue-152-subprocess-tune-plot → fix/issue-147-openmp-default-subprocess）
- **関連 Issue**: [#152](https://github.com/nbx-liz/LizyML-Widget/issues/152)（P-036 follow-up）
- **背景**:
  - P-036（PR #151）で Linux + libgomp 環境のデフォルト実行戦略を subprocess に切り替えた結果、`w.tune()`（先行 `w.fit()` なし）の直後に Tuning History（`optimization-history` plot）が "Loading plot..." で停止する回帰が発覚した。
  - 原因トレース（[issue #152 本文](https://github.com/nbx-liz/LizyML-Widget/issues/152)抜粋）:
    1. subprocess: `model.tune()` は auto-fit しないため、`adapter.export_model` 経由の `model.export()` が `MODEL_NOT_FIT` を投げる。
    2. `_subprocess_entry.py` が例外を silent swallow → `model_path = None` → 親プロセス `service._model = None` のまま。
    3. UI が `optimization-history` plot を要求 → `service.get_plot` が `ValueError("No trained model")` → `plot_error` 送信。
  - `tuning_plot()` は `_tuning_result`（`@dataclass(frozen=True)` の `TuningResult`）のみを参照し、`_study`（Optuna handle）は不要。`TuningResult` は frozen dataclass + 入れ子 dataclass のみで構成され完全に pickleable。
  - 同時に解決すべき UX バグ: `usePlot` の `plot_error` ハンドラは `loading[pt]` を解除するが、`PlotViewer` が独自 loading state を保持しており、エラー時にローディング表示が残り続ける。
- **提案内容**:
  - **(a) BackendAdapter Protocol に 2 メソッド追加**:
    ```python
    class BackendAdapter(Protocol[ModelT]):
        # ... existing ...
        def export_tune_state(self, model: ModelT, path: str) -> None: ...
        def restore_tune_state(self, model: ModelT, path: str) -> None: ...
    ```
    - `export_tune_state`: subprocess 内で `_tuning_result`（必須）と `_study`（best-effort、pickle 失敗時は省略）を pickle 形式で `path` に書き出す。
    - `restore_tune_state`: 親プロセスで `path` から読み戻し、freshly-created model に注入する。private slot への書き込みは adapter 内に閉じ込め、Widget / Service / 共通型は知らない。
  - **(b) IPC 経路の追加**: `_subprocess_entry.py` の tune ブランチで:
    1. tune-only では `adapter.export_model`（fit 前提）を **試行しない**。
    2. 代わりに `adapter.export_tune_state(model, tune_state_path)` を呼ぶ。
    3. `result_msg["tune_state_path"]` を同梱。
  - **(c) Service 拡張**: `WidgetService.restore_tune_state_from_path(path, *, config, df)` を追加。adapter 経由で空モデル + tune state を構築し、`self._model` にセット。`is_model_fitted` は False を維持。
  - **(d) JobRunner**: `SubprocessJobRunner` が `tune_state_path` を service に渡し、`finally` で確実に削除（`model_path` と対称）。
  - **(e) UI 修正**: `PlotViewer` が `plot_error` を受信したらローディング表示を確実に解除し、エラーメッセージを表示する。`usePlot` の cleanup 確認用 vitest を追加。
  - **(f) regression test**: `tests/regression/test_reg_152_subprocess_tune_plot.py` を追加（`LizyMLAdapter.tune` を mock、deterministic）。`5497eab`（PR #151 tip）で fail し、修正後に pass。
- **Invariants**:
  - INV-1: subprocess tune 完了時、`tune_state_path` は (a) `None` または (b) 親が読める path のいずれか。読めない/破損は `plot_error` に降格し widget 全体は壊さない。
  - INV-2: 親プロセスで `restore_tune_state` 後、`is_model_fitted(model) == False` を維持する。
  - INV-3: 親プロセスで `restore_tune_state` 後、`model._tuning_result is not None` であり、`available_plots(model)` が `optimization-history` を含む。
  - INV-4: tune state の export は `adapter.export_model` 試行より前に実施。export 失敗が tune state を巻き込まない。
  - INV-5: tune state ファイルは subprocess 終了かつ親側読込完了後に必ず削除される（`finally` cleanup）。
- **影響範囲**:
  - `src/lizyml_widget/adapter.py` — `BackendAdapter` Protocol に 2 メソッド宣言、`LizyMLAdapter` 実装
  - `src/lizyml_widget/_subprocess_entry.py` — tune ブランチで `export_tune_state` を呼び、`export_model` を tune-only でスキップ
  - `src/lizyml_widget/run_subprocess.py` — `SubprocessResult` に `tune_state_path` 追加
  - `src/lizyml_widget/job_runner.py` — `SubprocessJobRunner` で `restore_tune_state_from_path` 呼び出し + cleanup
  - `src/lizyml_widget/service.py` — `restore_tune_state_from_path` メソッド追加
  - `js/src/components/PlotViewer.tsx`, `js/src/hooks/usePlot.ts` — `plot_error` 時のローディング解除を保証
  - `tests/test_adapter_tune_state.py`, `tests/test_service_tune_state.py`, `tests/test_subprocess_integration.py`, `tests/regression/test_reg_152_subprocess_tune_plot.py` — 新規 / 拡張
  - `js/src/__tests__/PlotViewer.test.tsx`, `js/src/__tests__/usePlot.test.ts` — 拡張
  - `BLUEPRINT.md` §3.3 / §3.7.1 — Adapter Protocol API 追加と subprocess IPC ペイロード変更を記載
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]`
- **互換性**:
  - 公開 Python API（`w.fit()`, `w.tune()`, `w.retune()`, `w.apply_best_params()`）の signature / 振る舞いに変更なし。
  - `BackendAdapter` Protocol 拡張は **追加のみ** であり既存実装（`LizyMLAdapter`）以外を破壊しないが、新規 backend 実装者は 2 メソッドを実装する必要がある。
  - subprocess IPC `result_msg` に `tune_state_path` フィールドが追加される。古い subprocess バイナリと新しい parent の組み合わせでは `tune_state_path` が欠落するが、parent 側は missing を許容して従来挙動（plot 不可 + plot_error）にフォールバックする。
- **代替案（却下）**:
  - **案A: Tune を thread モードにフォールバック** — P-036 で fix した #147 の 30x 劣化を tune で再発させる。
  - **案B: subprocess で tune 後に auto-fit** — `tune()` の意味論変更（lizyml 仕様と乖離）、wall-clock が +1 fit 分悪化。retune（#128）への影響も大きい。
  - **案D: subprocess で plot を pre-render し plotly_json をキャッシュ** — 局所修正で短期的には簡潔だが、post-tune plot が増えるたびに同じ対応が必要。Optuna study handle が親に来ないため #128 の取り組みに使えない。
  - 案C を選んだ理由: **長期方向（#128 retune resume）と #152 の plot 復元を同じ仕組みで満たす**。Adapter Protocol 拡張のコストを払う代わりに、tune state を first-class IPC ペイロードとして扱う。
- **Optuna study の扱い**:
  - `_study` は InMemoryStorage（デフォルト）の場合 pickleable、RDBStorage の場合は不可。`export_tune_state` は `pickle.dumps(model._study)` を try/except で best-effort 実行し、失敗したら `study=None` で blob を出力する（warn ログのみ）。
  - 親側で復元された `_study` は #128 retune resume の前提となる。本 Proposal 自体は retune 対応を含まない（Issue #128 / #129 で別途）。
- **受け入れ基準**:
  - Linux + libgomp デフォルト環境で `w.tune()` 直後に Results → Tuning History が描画される。
  - `tests/regression/test_reg_152_subprocess_tune_plot.py` が `5497eab` で fail し、本 PR で pass する。
  - Adapter Protocol 拡張により新 backend 実装者は 2 メソッドの実装が必要であることを BLUEPRINT §3.3 に明記。
  - `_subprocess_entry.py` の tune-without-fit `model.export()` silent swallow が排除される。
  - `PlotViewer` が `plot_error` 受信時にローディング表示を解除し、明示的なエラーメッセージを出す（vitest で assert）。
  - 既存テストスイート（`uv run pytest`、`pnpm test:coverage`）は全 green を維持。

### P-036: libgomp 環境でのデフォルト実行戦略を subprocess に切り替え（OpenMP プール親和性回帰の根本対策）

- **日付**: 2026-05-09（提案・決定・実装）
- **ステータス**: 決定（実装済み — fix/issue-147-openmp-default-subprocess → develop）
- **関連 Issue**: [#147](https://github.com/nbx-liz/LizyML-Widget/issues/147)
- **背景**:
  - P-020 で導入された OpenMP プール親和性問題（GCC bug #108494, libgomp）の subprocess 回避策が、現状デフォルトで無効化されている。
  - [widget.py:571-581](https://github.com/nbx-liz/LizyML-Widget/blob/develop/src/lizyml_widget/widget.py#L571-L581) のゲートは `LZW_FORCE_SUBPROCESS=1` env var が設定されている場合にのみ `get_execution_strategy()` の戻り値を採用し、デフォルトでは無条件に `"thread"` を選ぶ。インラインコメントは劣化幅を「1.0–1.2x」と記載しているが、issue #147 で実測 30–35x の劣化が再現された（`worker_thread_fit / main_thread_fit = 30.4x`、worker thread 起動ごとに OS スレッドが ~30 本リーク）。
  - さらに `openmp_detect.is_libgomp_affected()` は `/proc/self/maps` を読むだけで lightgbm を import しないため、`LizyWidget.__init__` 時点では libgomp がまだロードされておらず、たとえ env var ゲートを外しても `("thread", None)` を返してしまう（検知が too lazy）。
  - Tune は Optuna trial ごとに OpenMP parallel region に再入するため、Fit 単発の劣化（30x）が trial 数倍に積算され、50 trial で end-to-end 20–50x 劣化として観測される。
- **検証済み再現**:
  - reproducer (`tests/regression/test_reg_147_openmp_perf.py` のもとデータ): `main: 0.16s` → `worker: 4.77s` (`30.4x`), threads 94 → 126。
  - learned skills `openmp-daemon-thread-degradation`, `openmp-thread-pool-accumulation` が示す挙動と一致。
- **提案内容**:
  - **(a) 検知の deferred 化**: `is_libgomp_affected()` を呼び出し時に lightgbm を force-import してから `/proc/self/maps` を読むよう変更し、結果を module-level cache に保存する（同一プロセス内で結果は不変）。`get_execution_strategy()` を最初に呼ぶ場面（最初の Fit/Tune 直前）で正しい判定が出るようになる。
  - **(b) ゲートの極性反転**: `LZW_FORCE_SUBPROCESS=1` ゲートを廃止し、libgomp が検知されたら subprocess をデフォルト戦略にする。debug 等で in-process 実行が必要な場合は `LZW_FORCE_THREAD=1` で opt-out できるようにする。
  - **(c) インラインコメント修正**: `widget.py` の "1.0-1.2x" の文言を実測値（fit 30x / tune 20–50x）に書き換える。
  - **(d) regression test**: `tests/regression/test_reg_147_openmp_perf.py`（`pytest.mark.slow`）を追加し、reproducer 相当の workload で `worker / main < 2.0x` を assert する。CI は default suite に含めない（slow 系は ad-hoc / nightly）。
- **Invariants**:
  - INV-1: `LZW_FORCE_THREAD=1` が設定されている場合、`_execution_strategy == "thread"` （`get_execution_strategy()` の戻り値に依らず）。
  - INV-2: `LZW_FORCE_THREAD` 未設定かつ libgomp 検知 = True なら `_execution_strategy == "subprocess"`。libgomp 未検知（macOS / Windows / libomp 環境）なら `"thread"`。
  - INV-3: `is_libgomp_affected()` の結果は同一プロセス内で冪等（cache 後に変化しない）。
- **影響範囲**:
  - `src/lizyml_widget/openmp_detect.py` — `is_libgomp_affected()` に lightgbm force-import + cache 追加
  - `src/lizyml_widget/widget.py` §`_run_job` — env var ゲートを反転、コメント更新
  - `tests/test_openmp_detect.py` — 新規（detection lifecycle / cache テスト）
  - `tests/regression/test_reg_147_openmp_perf.py` — 新規（`pytest.mark.slow`、wall-clock regression）
  - `BLUEPRINT.md` §3.7 — 新デフォルト戦略と opt-out env var を文書化
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]`
- **互換性**:
  - 公開 Python API（`w.fit()`, `w.tune()`, `w.retune()`）の signature / behaviour は変わらない。subprocess 経路は P-020 で既に実装済みで、retune 以外のすべての job type をサポートする。
  - **挙動変更**: Linux + libgomp 環境では Fit/Tune がデフォルトで subprocess 実行となるため、Python ログの一部が parent-process 側で捕捉されない可能性がある（既存の subprocess runner は stdout/stderr を pipe して進捗イベントを抽出する設計）。
  - **opt-out**: 旧挙動を維持したいユーザーは `LZW_FORCE_THREAD=1` を設定する。`LZW_FORCE_SUBPROCESS=1` は未指定でも subprocess になるため redundant となるが、後方互換のため warning なく無視する。
  - **retune**: P-020 当時から subprocess runner は retune 未対応 (`RetuneSubprocessUnsupportedError`)。本 Proposal はその制約に変更を加えない。retune は依然 thread 実行となる（issue #128 で別途扱う）。
- **代替案（却下）**:
  - **案A: env var ゲートをそのまま残し、デフォルト thread を維持** — 30x の劣化を放置することになり、issue #147 の根治にならない。learned skill に「numeric claim を test で pin する」原則を加えた直後に再発させた経緯（PR #117/#144 で gate を維持した）も踏まえ、デフォルト挙動の修正が妥当。
  - **案B: lightgbm 自体を libomp build に置き換える** — ユーザー環境のビルドに侵入しなければならず、apt 配布の lightgbm では非現実的。LD_PRELOAD は subprocess 経路で既に提供済みで、デフォルト subprocess 化のほうが副作用が小さい。
  - **案C: thread 実行のままで `omp_set_num_threads()` / `threadpoolctl` を強制** — P-020 の検証済みアプローチで効果なしと判明済み（プール親和性は ICV では解消できない）。
- **受け入れ基準**:
  - Linux + libgomp 環境のデフォルト `w.tune()` が subprocess で実行され、`htop` で main thread と同程度のコア使用率を観測できる。
  - `LZW_FORCE_THREAD=1` で旧 thread 経路に切り替え可能。
  - `tests/regression/test_reg_147_openmp_perf.py` (`pytest.mark.slow`) が `worker/main < 2.0x` を assert し、ローカルで pass する。
  - `tests/test_openmp_detect.py` で (i) `lightgbm` 未 import 状態と import 後で結果が変化すること、(ii) cache が安定していることを assert する。
  - `widget.py` のインラインコメントが実測値（30x / 50 trial で 20–50x）に更新される。
  - `BLUEPRINT.md` §3.7 に新デフォルトと opt-out env var が記載される。
  - 既存テストスイート（`uv run pytest`、`pnpm test:coverage`）は全 green を維持。

### P-035: TuningSummary に config_snapshot / ui_snapshot を追加

- **日付**: 2026-05-08（提案）/ 2026-05-09（決定・実装）
- **ステータス**: 決定（実装済み — PR #132 → develop）
- **関連 Issue**: [#132](https://github.com/nbx-liz/LizyML-Widget/issues/132)
- **背景**:
  - 現状、`LizyWidget` は `_tune_config_snapshot` / `_tune_ui_snapshot` の 2 個の
    private 属性を保持し、`_run_job("tune")` 開始時に当時の `full_config` と `self.config`
    のコピーを書き込む（[widget.py:961-963](https://github.com/nbx-liz/LizyML-Widget/blob/develop/src/lizyml_widget/widget.py#L961-L963)）。
  - これらは後の `_handle_apply_best_params` で `service.apply_best_params(..., tune_snapshot=...,
    tune_ui_snapshot=...)` に渡される（[widget.py:823-840](https://github.com/nbx-liz/LizyML-Widget/blob/develop/src/lizyml_widget/widget.py#L823-L840)）。
  - 結果として Widget が "ジョブ実行時の時系列スナップショット" を保持しており、
    CLAUDE.md §4 が定義する Widget の責務（traitlets 定義 / Action 処理 / スレッド管理のみ）に違反する。
    また `tune` と `apply_best_params` の間に `load()` / `set_target()` 等が走るとスナップショットが
    silently stale になり、apply 結果が誤った config を生成する隠れた時系列依存が発生する。
- **提案内容**:
  - 共通型 `TuningSummary`（`src/lizyml_widget/types.py`）に 2 つの読み取り専用フィールドを追加:
    ```python
    @dataclass(frozen=True)
    class TuningSummary:
        # ... existing fields ...
        config_snapshot: Mapping[str, Any] = field(default_factory=dict)  # canonical full_config
        ui_snapshot: Mapping[str, Any] = field(default_factory=dict)      # widget-side ui config
    ```
  - `BackendAdapter.tune` / `Adapter.tune` は `TuningSummary(config_snapshot=...,
    ui_snapshot=...)` を返すよう更新する（adapter は呼び出し元から受け取った canonical config と
    ui config を Result に閉じ込める）。
  - `WidgetService.tune` は `_tune_model` だけでなく最新 `TuningSummary` も保持し、
    `apply_best_params` は `tune_snapshot=...` / `tune_ui_snapshot=...` の引数を取らず、
    Service が保持する最新 summary から読み出す。
  - `LizyWidget` から `_tune_config_snapshot` / `_tune_ui_snapshot` を削除。
- **影響範囲**:
  - `src/lizyml_widget/types.py` — `TuningSummary` に 2 フィールド追加（共通型変更：change gate）
  - `src/lizyml_widget/adapter.py` — `tune` 結果に snapshot を埋める
  - `src/lizyml_widget/service.py` — `apply_best_params` のシグネチャ変更（snapshot 引数削除）
  - `src/lizyml_widget/widget.py` — `_tune_*_snapshot` 属性削除、`_handle_apply_best_params` 簡素化
  - `BLUEPRINT.md` §3.2 / §6 — 共通型と Widget 責務の対応更新
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]`
  - `tests/test_widget_actions.py`, `tests/test_service*.py` — `apply_best_params` のシグネチャに合わせて更新
- **互換性**:
  - 公開 Python API（`w.tune()`, `w.apply_best_params(...)`）は変わらない。
  - `TuningSummary` は dataclass で `default_factory=dict` を持つため、既存テストが
    construct する `TuningSummary(...)` は引数を増やさなくても通る（後方互換）。
  - `Service.apply_best_params` の `tune_snapshot=` / `tune_ui_snapshot=` 引数を削除するのは
    Widget 内部の呼び出し元のみ（外部公開なし）。
- **代替案（却下）**:
  - **案A: snapshot は Widget 側に保持したままにする** — CLAUDE.md §4 の Widget 責務違反を
    残し、tune/apply 間の競合バグの根本原因が解消されない。
  - **案B: snapshot を `_tune_model` の attribute として lizyml model に書き込む**
    — `model._widget_config` で過去に同種の私有書き込みを禁じた経緯（#116）がある。
    Adapter の id-based registry に置く案も検討したが、TuningSummary（既に DTO として存在）に
    含めるほうが意味論的に正しい。
- **受け入れ基準**:
  - `TuningSummary` に `config_snapshot` / `ui_snapshot` フィールドが追加され、Adapter が埋める。
  - `LizyWidget._tune_config_snapshot` / `_tune_ui_snapshot` が削除される。
  - `Service.apply_best_params(params, current_config)` のシグネチャは外部から見て不変
    （内部実装が `_last_tune_summary` から snapshot を取り出す）。
  - 新規テスト: `apply_best_params` 直前に `load()` が走った場合に、stale な snapshot から
    再構築するのではなく明示エラー（"tune summary cleared by load"）を返す。
  - 既存 `tests/test_widget_actions.py::test_apply_best_params_*` が引き続き green。
  - `mypy --strict` 通過。
- **実装ノート（2026-05-09）**:
  - `TuningSummary` に `config_snapshot` / `ui_snapshot` を追加（dataclass のため
    `default_factory=dict` で後方互換）。
  - `WidgetService.tune` は新しく `ui_snapshot=` kwarg を受け取り、Adapter 結果を
    `dataclasses.replace` で `config_snapshot` / `ui_snapshot` 入りに差し替えて
    `_last_tune_summary` に格納する（Service 内 lock 配下）。
  - `Service.apply_best_params` は `tune_snapshot=` / `tune_ui_snapshot=` の kwarg を
    完全削除。`_last_tune_summary` がなければ `current_config` ベース、
    `load_data` 後の stale 状態では `ValueError("tune summary cleared by load …")` を発火。
  - `JobSpec.ui_snapshot` を新設し、Widget の `_run_job` で
    `copy.deepcopy(dict(self.config))` を tune ジョブのみ詰める。
  - `SubprocessJobRunner` は `service.record_subprocess_tune_summary(...)` 経由で
    親プロセス側 Service に TuningSummary を再構築する（subprocess 経路は別 process なので
    `service.tune` 側で記録できないため）。
  - `LizyWidget.__init__` から `_tune_config_snapshot` / `_tune_ui_snapshot` を削除、
    `WidgetActionDispatcher.handle_apply_best_params` も簡素化（snapshot 受け渡しなし）。

---

### P-034: BackendContract に UI defaults / step_map を追加

- **日付**: 2026-05-08（提案）/ 2026-05-09（決定・実装）
- **ステータス**: 決定（実装済み — PR #131 → develop）
- **関連 Issue**: [#131](https://github.com/nbx-liz/LizyML-Widget/issues/131)
- **背景**:
  - PR #119 / #121 で JS から backend 固有 option set / parameter catalog をハードコードする
    箇所をほぼ撤廃したが、**数値デフォルト** と **step 値** は対象外だった:
    - `DataTab.tsx`: `n_splits ?? 5`, `random_state ?? 42`, `gap ?? 0`, `purge_gap ?? 0`,
      `embargo ?? 0`
    - `TuneSubTab.tsx` / `SearchSpace.tsx`: `n_trials ?? 10`, `_precision_at_k_k ?? 10`
    - `ModelEditors.tsx`, `SearchSpace.tsx`: feature weight `step={0.1}`
    - `RetuneControls.tsx`: `boundary_threshold` `step={0.01}`
  - これらは "payload 未設定時のフォールバック" として `??` 演算子で書かれているが、
    backend 仕様（lizyml）が変わると JS 側のコード変更なしには反映されない。
    CLAUDE.md §8 の "JS に backend 固有値をハードコードしない" 原則を厳密に守ると違反となる。
- **提案内容**:
  - `BackendContract.ui_schema`（`src/lizyml_widget/adapter_contract.py`）に 2 つの dict を追加:
    ```python
    ui_schema["defaults"] = {
        "cv": {
            "n_splits": 5, "random_state": 42, "gap": 0,
            "purge_gap": 0, "embargo": 0,
        },
        "tune": {"n_trials": 10},
        "metric_params": {"precision_at_k_k": 10},
    }
    ui_schema["step_map"] = {
        # 既存エントリに加えて:
        "feature_weights": 0.1,
        "boundary_threshold": 0.01,
    }
    ```
  - JS 側で `??` リテラルフォールバックを撤廃し、`capabilities.ui_schema.defaults.cv.n_splits`
    / `step_map.feature_weights` 等を look-up する。
- **影響範囲**:
  - `src/lizyml_widget/adapter_schema.py` または `adapter_contract.py` — `defaults` / `step_map` 拡張
    （**change gate**: data contract / backend contract 変更）
  - `js/src/tabs/DataTab.tsx` / `TuneSubTab.tsx` / `components/SearchSpace.tsx`
    / `ModelEditors.tsx` / `RetuneControls.tsx` — `??` フォールバック削除
  - `tests/test_adapter_contract.py` 等 — contract shape の golden test 追加
  - `js/src/__tests__/` — contract-driven レンダリングのケース追加
  - `BLUEPRINT.md` §3.2 — backend contract 形状の更新
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]`
- **互換性**:
  - `BackendContract` schema_version は据え置き（後方互換な追加のみ）。
  - 既存の `step_map` キーには手を入れず、`feature_weights` / `boundary_threshold` を**追加**するのみ。
  - 新規 `defaults` キーは optional：JS 側はキー欠落時に既存の `??` 値（互換のため一時保持）を使う
    実装にする → 移行完了 PR で `??` を全削除。
- **代替案（却下）**:
  - **案A: 数値リテラル程度なら JS にハードコードしてよい** — 既存 PR #119 の方針と矛盾し、
    "backend 固有値は contract 由来" の不変条件を弱める。
  - **案B: 個別の traitlet で defaults を渡す** — 既に `backend_contract` 経由で
    capabilities が流れているため、新 traitlet を増やすと sync 経路が増えて煩雑になる。
- **受け入れ基準**:
  - `BackendContract.ui_schema.defaults` と `step_map` の追加分が adapter から流れ、
    `tests/test_adapter_contract.py`（または `test_frontend_contract.py`）の golden test で
    shape が固定される。
  - JS 側の `?? 5`, `?? 10`, `?? 42`, `step={0.1}`, `step={0.01}` リテラルが Issue に列挙された
    すべてのファイルから消える（grep で残存 0）。
  - `pnpm test:coverage` の vitest threshold（statements 75% / branches 70%）を維持。
  - `js/src/__tests__/` で contract-driven レンダリング経路を最低 1 ケースずつ検証。
- **実装ノート（2026-05-09）**:
  - `adapter_contract.build_ui_schema` の `defaults` に `cv` / `tune` / `metric_params` を、
    `step_map` に `feature_weights` / `boundary_threshold` を追加。`schema_version` は据え置き。
  - JS では contract から読み出すための薄いヘルパ（`dN(key, fallback)` 等）を追加し、
    fallback 数値は **fixture が contract を渡さないユニットテスト用のみ** 残した。
  - 影響ファイル: `DataTab.tsx` / `TuneSubTab.tsx` / `SearchSpace.tsx` / `ModelEditors.tsx`
    （`metricParamDefaults` prop を ModelSection→TypedParamsEditor で伝搬） /
    `RetuneControls.tsx` / `ResultsTab.tsx` / `App.tsx`（`stepMap` を contract から RetuneControls へ伝搬）。
  - ゴールデンテスト: `TestContractNumericDefaultsAndStepMap` を `tests/test_frontend_contract.py` に追加。
  - 既存 vitest 274 件 / pytest 935 件すべて green、ruff + mypy strict クリーン。

---

### P-033: 状態機械不変条件（INV-A..F）の宣言と enforcement

- **日付**: 2026-05-08
- **ステータス**: 提案
- **関連 Issue**: [#118](https://github.com/nbx-liz/LizyML-Widget/issues/118)
- **背景**:
  - `~/.claude/rules/common/invariants-first.md` は並行性 / 状態機械 / リソース所有権を扱うコードに対して、
    実装前に不変条件を宣言し、テストで enforce することを義務付けている。
  - 本リポジトリの状態機械（`status` traitlet, `_job_thread`, `WidgetService._tune_model`, `_cancel_flag`,
    `progress.round`, `tune_summary.boundary_report.dims`）には、これまで形式化された不変条件が存在しなかった
    （`grep -r "INV-" docs/ src/` で 0 ヒット）。
  - P-027 / P-028 / P-029 で発生したラウンド +1 ずれ等の状態機械バグは、不変条件が文書化されていなかったため
    fix が "発見した症状" に対するパッチに留まり、累積的な負債となった（HISTORY.md Bug Fix 参照）。
  - P-032（issue #117）で `_supervise` に状態遷移ロジックが集約されたため、ここで invariant assert を
    encode する好機となる。
- **提案内容**:
  - **BLUEPRINT.md §6 に「State machine invariants」節（§6.4）を追加** し、INV-A..F を以下の形式で宣言:
    - **INV-A**: `status` の遷移は `idle → data_loaded → running → {completed | failed} → running → ...`
      の有限状態機械に従う。`idle → completed` 等の不正遷移は即座に拒否される。
    - **INV-B**: `_job_thread` は同時に最大 1 個のライブワーカーのみを保持する。
      `status == "running"` 中の `_run_job` 再呼び出しは黙って無視される（`widget.py::_run_job` の
      ガード）。
    - **INV-C**: `WidgetService._tune_model` は最直近の `tune` 呼び出しが所有する。
      `tune` → `fit` → `retune` の順序で fit が `_tune_model` を破壊しない（P-028）。
    - **INV-D**: `_cancel_flag` は各ジョブ開始時に `clear()` され、ジョブ終了まで Widget 側からは
      書き込まれない。`status == "running"` 中の cancel は `failed` (`error.code == "CANCELLED"`)
      へ遷移する。
    - **INV-E**: `progress.round` は単一の tune 呼び出し（resume 含む）内で単調非減少。
      P-029 のラウンド +1 オフバイワンに対するレギュラリゼーション。
    - **INV-F**: `tune_summary.boundary_report.dims` は各 search space 次元を **過不足なく一度ずつ**
      列挙する（重複・欠損なし）。ラウンドごとの差分ではなく累積スナップショット。
  - **invariant tests を新規 `tests/test_invariants.py` に集約**:
    - INV-A: 不正遷移 (`idle → completed`) を試み、`status` が変わらないこと。
    - INV-B: `status == "running"` 中の `_run_job` 再呼び出しが `_job_counter` を増やさないこと。
    - INV-C: `tune → fit → retune` シーケンスで `_tune_model` が破壊されない。
    - INV-D: `_cancel_flag` が新規ジョブ開始時に `False` になる。Cancel 中の状態遷移が `failed` で
      `error.code == "CANCELLED"` になる。
    - INV-E: round が monotonic（連続した round 値が降順にならない）。
    - INV-F: `boundary_report.dims` のすべての `name` が unique で、search space 全次元と一致する。
- **影響範囲**:
  - `BLUEPRINT.md` §6.4 — 新規節（**change gate**: invariants over shared state）
  - `tests/test_invariants.py` — 新規（INV-A..F 検証）
  - `src/lizyml_widget/widget.py` — runtime guard の comment 整備（INV-X breadcrumbs）
  - `HISTORY.md` — 本 Proposal
  - `CHANGELOG.md` — `[Unreleased]` セクション
- **互換性**:
  - 公開 Python API / traitlet / JS dispatcher 互換性は変更なし。
  - 内部 assert / log は debug-level；ユーザーから観測可能な動作変化なし。
- **代替案（却下）**:
  - **案A: 不変条件はコメントのみで十分** — invariants-first.md は "executable checks > comments" を
    明示。コメントは書き手のメンタルモデルしか保存しない。
  - **案B: assert ではなく型レベルで強制（ブランド型）** — Python の型システムで状態機械を
    完全エンコードするには `typing.Literal[...]` の transition 型が必要となり、コードが煩雑化する。
    test + runtime assert で十分。
- **受け入れ基準**:
  - BLUEPRINT.md §6.4 に INV-A..F が `INV-N: <subject> ... — violated if <scenario>` 形式で宣言される。
  - `tests/test_invariants.py` で INV-A..F 各々に対応する RED-then-GREEN テストが存在する。
  - `_supervise` / `_run_job` の状態遷移ガードに対応する INV-X breadcrumb がある。
  - 既存テスト全 green。
  - 今後の `_supervise` / `status` / `_tune_model` / `_cancel_flag` を触る PR は body に
    Invariants + Failure Paths セクションを含める運用が確立する。

---

### P-032: JobRunner Protocol 抽出（widget.py God-class 分割）

- **日付**: 2026-05-08
- **ステータス**: 提案
- **関連 Issue**: [#117](https://github.com/nbx-liz/LizyML-Widget/issues/117)
- **背景**:
  - `src/lizyml_widget/widget.py` は現在 1234 行（CLAUDE.md §8 の 800 行上限を超過）。
    Python API surface（`load` / `fit` / `tune` / `set_config` / properties）と、
    JS-action dispatcher + thread orchestrator + dual job worker
    （`_run_job` / `_job_worker` / `_subprocess_job_worker`）を融合している。
  - `_job_worker` と `_subprocess_job_worker` は state 遷移・error 分類・traitlet 配信を
    near-duplicate で実装しており、再 tune 機能（P-028）は subprocess 経路でのフォーク維持コストが
    高すぎたため subprocess 実行が **disable** されたままになっている（issue 本文参照）。
  - 状態機械が複数箇所に分散しているため、INV-A..F（issue #118 で宣言予定）を後続で書く際にも
    enforcement が困難。
- **提案内容**:
  - 新モジュール `src/lizyml_widget/job_runner.py` を作成し、以下を定義:
    - `JobRunner` Protocol — `run(spec, on_progress, cancel_event) -> JobResult` の単一メソッド。
    - `ThreadJobRunner` — 既存 `_job_worker` のロジックを移植したインプロセス実装。
    - `SubprocessJobRunner` — 既存 `_subprocess_job_worker` のロジックを移植したアウトプロセス実装。
  - `JobSpec` / `JobResult` を共通 dataclass として導入（job_type, config, retune_kwargs, etc.）。
  - `widget.py::_run_job` は薄い supervisor に統合:
    - state 遷移（status, job_type, job_index, progress, elapsed_sec, error, fit_summary, tune_summary）
    - error classification（BACKEND_ERROR / INTERNAL_ERROR）
    - traitlet 配信
    - cancel flag のリセット

    上記は単一の `_supervise(runner, spec)` 内で実装し、両 runner で共有する。
  - 既存の `_job_worker` / `_subprocess_job_worker` を削除し、widget.py を 800 行未満に圧縮。
  - 再 tune の subprocess 経路 enable は別 follow-up（model artifact ハンドオフが独立した設計を要する）。
    本 Proposal の範囲では `RETUNE_SUBPROCESS_UNSUPPORTED` ガードを `SubprocessJobRunner` に移植するのみ。
- **影響範囲**:
  - `src/lizyml_widget/widget.py` — `_run_job` 簡略化、`_job_worker` / `_subprocess_job_worker` 削除（**change gate**: 並行性 / 所有権設計）
  - `src/lizyml_widget/job_runner.py` — 新規（**change gate**: 共通型 `JobSpec` / `JobResult` 追加）
  - `src/lizyml_widget/subprocess_runner.py` — `SubprocessJobRunner` への移植先として参照
  - 既存 `tests/test_widget_jobs.py` / `tests/test_subprocess_integration.py` / `tests/test_thread_safety.py` —
    新 API で再構成
  - 新規 `tests/test_job_runner.py` — runner 単体（normal completion / cancel mid-run / exception mid-run）
  - `CHANGELOG.md` — `[Unreleased]` セクション
  - `HISTORY.md` — 本 Proposal
- **互換性**:
  - 公開 Python API（`LizyWidget.fit` / `tune` / `retune` / `load` / `predict` 等）の振る舞いは変更なし。
  - 公開 traitlet（`status` / `job_type` / `job_index` / `progress` / `elapsed_sec` / `error` /
    `fit_summary` / `tune_summary` / `available_plots` / `inference_result`）の意味論は変更なし。
  - JS 側 action dispatcher の入出力契約は変更なし。
  - `LZW_FORCE_SUBPROCESS=1` 環境変数の挙動は維持。
- **代替案（却下）**:
  - **案A: widget.py 内で `_job_worker` と `_subprocess_job_worker` の共通ロジックを helper に括り出すだけ** —
    state machine が依然 widget.py に残るため、INV-A..F の宣言・enforcement と並行性設計の単一責任に矛盾する。
    Issue #118 の前提を崩す。
  - **案B: subprocess 実行をデフォルトに切り替える** — OpenMP 関連の現実的な fit 劣化は 1.0–1.2x で、
    subprocess 起動 overhead（≈ 500ms import）の方がレイテンシ影響が大きい。デフォルト切り替えは
    P-032 の範囲外。
  - **案C: 抽出をやめて 1234 行の現状を許容** — CLAUDE.md §8（God-class 禁止）違反で、
    今後の機能追加（async runner / WebWorker 等）に対する拡張コストが累積する。
- **受け入れ基準**:
  - `src/lizyml_widget/job_runner.py` が存在し、`JobRunner` Protocol + `ThreadJobRunner` +
    `SubprocessJobRunner` を提供する。
  - `widget.py` は 800 行未満。
  - `widget.py` 内の state 遷移ロジックは `_supervise` のみが管理する（他の `self.status =` 代入は
    Python API の data_loaded セットなど job 外の用途のみ）。
  - 新規 `tests/test_job_runner.py` で各 runner について正常完了 / cancel / 例外を assert。
  - 既存 898 Python テスト + 272 JS テスト + 24 e2e テストが green。
  - `ruff check` / `ruff format --check` / `mypy --strict` / `pytest` / `eslint` / `tsc` / `vitest` 全 green。

---

### P-030: lizyml 0.10 / 0.11 / 0.12 互換窓拡大

- **日付**: 2026-05-08
- **ステータス**: 承認・実装
- **関連 Issue**: [#112](https://github.com/nbx-liz/LizyML-Widget/issues/112)
- **決定事項**:
  - **Phase 1–3 を実装**: 互換窓拡大 (`>=0.10.0,<0.13`) + 0.10 の非数値ラベル分類の通過確認 +
    0.11 の `smape` / `wape` 回帰メトリックの BackendContract 露出。
  - **Phase 4 (0.12 resumable tuning) は本 Proposal では UI 露出しない**: `Tuner.tune(storage=, study_name=)`
    の Widget 経由公開は SQLite path lifecycle / kernel 横断 study 共有 / cleanup の独立した設計を要するため、
    将来の P-031（仮）で扱う。0.12 を compat 範囲に含めること自体の動作確認は Phase 1 + Phase 5 の smoke で
    完了している。
  - **Widget 0.9.0 として release**: lizyml 0.9.x は compat 範囲から外れるため、minor bump とする。
- **背景**:
  - LizyML が 0.9.0 公開後に `0.9.1` / `0.10.0` / `0.11.0` / `0.12.0` を続けて release。
  - lizyml-widget 0.8.0 は `lizyml>=0.9.0,<0.10` に固定されており、`pyproject.toml` の extra と
    `src/lizyml_widget/adapter.py` の `LIZYML_MIN_VERSION` / `LIZYML_MAX_VERSION` が二重に
    範囲を強制するため、ユーザー側で `lizyml>=0.10` を選択できない。
  - 0.10–0.12 の主な変更は **加算的（破壊的変更なし）**:
    - **0.10**: 非数値分類ラベル自動エンコード（`FitResult.target_encoder` 新設、`FORMAT_VERSION` 1→2、
      旧 v1 artifact は `Model.load()` が in-memory migrate する）、新エラーコード `TARGET_NOT_NUMERIC` /
      `TARGET_UNSEEN_LABEL`。
    - **0.11**: 回帰メトリック `smape` / `wape` を `MetricRegistry` に追加、LightGBM feval bridge 経由で
      `eval_history` / 学習曲線にも反映。
    - **0.12**: Optuna 永続化ストレージによる resumable tuning（`Tuner.tune(storage=, study_name=)` 追加、
      `storage=None` で従来挙動）。
- **提案内容**:
  - **Phase 1 — 互換窓拡大（基盤）**:
    - `pyproject.toml` の lizyml extras を `>=0.10.0,<0.13` に更新（**lower bound を 0.10 に切り上げ**: 0.10
      の `target_encoder` 経由のラベル dtype 保持を Widget 側でも前提とするため）。
    - `LIZYML_MIN_VERSION = (0, 10, 0)`, `LIZYML_MAX_VERSION = (0, 13, 0)` に更新。
    - `tests/test_retune_monitoring.py` の guard 定数アサートを追従。
    - `docs/VERSION_COMPAT.md` に新行追加（`lizyml-widget 0.9.x` ↔ `lizyml >=0.10.0,<0.13`）。
    - `uv.lock` を `uv lock --upgrade-package lizyml` で再生成。
  - **Phase 2 — 0.10 統合**:
    - `LizyMLAdapter.predict()` は `result.pred` がすでに元 dtype（object / category / bool）でデコード済み
      な前提で受け取り、pandas DataFrame に dtype を保持して載せ替えるだけに留める。
    - 新エラーコード `TARGET_NOT_NUMERIC` / `TARGET_UNSEEN_LABEL` は `LizyMLError` の message に含めて
      `BACKEND_ERROR` 配下で表示する（Widget 側で再分類はしない、UI 上はメッセージで識別）。
    - 旧 v1 artifact の `Model.load()` 後方互換は lizyml 側で吸収済みのため、Widget 追加実装は不要。
      回帰テストでのみ verify する。
  - **Phase 3 — 0.11 統合**:
    - `adapter_params.py` の回帰タスク metric option set に `smape` / `wape` を追加。
    - `adapter_schema.py` の Search Space catalog（regression metric 選択肢）に `smape` / `wape` を加える。
    - `LizyMLAdapter.plot()` の `learning-curve` 経路は既存 `metrics` kwarg 透過で動作（追加実装不要、
      e2e テストで検証）。
  - **Phase 4 — 0.12 決定**:
    - **本 Proposal の範囲では UI 露出しない**。`Tuner.tune(storage=, study_name=)` を Widget で公開する場合
      SQLite ファイル lifecycle / kernel 横断 study 共有 / cleanup の設計が必要となるため、別 Proposal
      （P-031 仮）で扱う。compat 範囲に 0.12 を含めること自体の動作確認は Phase 1 + Phase 5 の smoke で
      カバーする。
- **影響範囲**:
  - `pyproject.toml` — extras / dev dependency 範囲（**change gate**: 外部依存変更）
  - `uv.lock` — 再生成
  - `src/lizyml_widget/adapter.py` — `LIZYML_MIN_VERSION` / `LIZYML_MAX_VERSION` 定数（**change gate**:
    BackendAdapter Protocol の依存）
  - `src/lizyml_widget/adapter_params.py` — regression metric option set
  - `src/lizyml_widget/adapter_schema.py` — Search Space catalog の regression metric
  - `tests/test_retune_monitoring.py` — guard 定数アサート
  - `tests/regression/test_reg_112_target_encoder_roundtrip.py` — 新規（非数値ラベル分類の round-trip）
  - `tests/regression/test_reg_112_smape_wape.py` — 新規（回帰メトリック登録 + learning curve）
  - `docs/VERSION_COMPAT.md` — 新行
  - `CHANGELOG.md` — `[0.9.0]` セクション
  - `HISTORY.md` — 本 Proposal
- **互換性**:
  - **Widget 0.9.0 リリース時、lizyml 0.9.x ユーザーは強制的に 0.10 へアップグレードが必要**。
    0.9.x の `Model.fit` / `Model.tune` / `Model.predict` 公開 API は 0.10 で互換維持されているため、
    アップグレード自体は import 互換のはず（ただし `FORMAT_VERSION` バンプにより lizyml 0.9 で保存した
    `model.lizyml` artifact を 0.9 で再ロードする経路は 0.10 への移行で 1 度だけ migrate される）。
  - `BackendAdapter` Protocol のシグネチャは無変更。
  - `FitSummary` / `TuningSummary` / `PredictionSummary` / `BackendInfo` の構造は無変更（0.10 の
    `target_encoder` フィールドは Adapter 内部で吸収、Widget の共通型には漏らさない）。
  - 既存 traitlet / action / Python API の互換破壊なし。
  - JS 側の backend_contract 駆動 UI は無変更（metric option set は backend 経由で配信）。
- **代替案（却下）**:
  - **案A: 0.10 / 0.11 / 0.12 を別々の PR / Proposal に分割** — 加算的変更が中心で blast radius が小さい
    ため、3 PR 分の CI / レビューコストに見合わない。単一 PR で段階コミットする方が churn が少ない。
  - **案B: 互換窓を `>=0.9.0,<0.13` のままにして 0.9 ユーザーを残す** — 0.10 の `target_encoder` を
    Adapter 内で前提として書くと 0.9 環境では型・属性が存在せず実行時に失敗する。条件分岐で吸収する
    複雑さは長期保守負債になるため不採用。
  - **案C: 0.12 の resumable tuning を本 PR で UI 露出する** — SQLite path 解決 / kernel restart 横断 /
    Widget 間 study 共有 / cleanup は独立した設計が必要。本 Proposal の主目的（compat 窓拡大）から外して
    P-031（仮）に分離する。
- **受け入れ基準**:
  - `pip install "lizyml-widget[lizyml]"` で `lizyml==0.12.x` が解決される。
  - `LizyWidget` の import / Fit / Tune / Inference が `lizyml==0.12.0` で全 green
    （Python 3.10 / 3.11 / 3.12 マトリクス CI）。
  - 非数値ラベル分類（`y` が `object` / `category` / `bool`）で `LizyWidget.fit()` → `predict()` が
    元 dtype を保持して結果を返す（回帰テストで assert）。
  - 回帰タスクで `metric=["smape", "wape"]` が backend に届き、学習曲線にプロットされる
    （e2e テストで assert）。
  - 旧 lizyml 0.9.x で保存された `model.lizyml` artifact が `Model.load()` 経由で読み込めることを
    smoke テストで確認（fixture 整備）。
  - `ruff check` / `ruff format --check` / `mypy --strict` / `pytest` / `eslint` / `tsc` / `vitest` 全 green。
  - `docs/VERSION_COMPAT.md` の対応表最上行が `lizyml-widget 0.9.x ↔ lizyml >=0.10.0,<0.13` になっている。

---

### Bug Fix: Convergence Signal の Round 表示 +1 ずれ + チェックマーク escape 不正

- **日付**: 2026-04-12
- **ステータス**: 承認・実装
- **種別**: バグ修正（change gate 対象外）
- **症状**: 3 回目の tune（= `w.tune()` + `w.retune()` × 2）完了後に Convergence Signal が
  以下のように表示されていた:
  - `\u2713` が 6 文字の文字列としてそのままレンダリングされる（チェックマーク glyph にならない）
  - 「Round 4 finished without expanding any boundary」と表示（本来は「Round 3」）
- **原因**:
  1. [js/src/components/ConvergenceSignal.tsx](js/src/components/ConvergenceSignal.tsx):
     JSX のテキストノードに `\u2713` を生書きしていた。JSX テキストは JavaScript 文字列リテラルではないため、
     バックスラッシュエスケープは解釈されずそのまま文字列として DOM に流れる。
     `{"\u2713"}` のように JSX 式で囲むか、`&#x2713;` の HTML entity を使う必要がある。
  2. [js/src/tabs/ResultsTab.tsx](js/src/tabs/ResultsTab.tsx):
     `lizyml.core.types.tuning_result.RoundSummary.round` は lizyml の docstring で
     **「1-indexed」と明記** されているにもかかわらず、UI 側で `lastRound.round + 1` と
     誤って +1 していた（P-027 実装時の誤解）。
     さらに `lastRound.round >= 1` の冗長な guard も残っていた（1-indexed なので常に真）。
- **修正内容**:
  - `ConvergenceSignal.tsx` L33: `\u2713` → `{"\u2713"}`
  - `ResultsTab.tsx` Convergence Signal 呼び出し:
    - `round={lastRound.round + 1}` → `round={lastRound.round}`
    - `lastRound.round >= 1` の冗長 guard を削除
    - `RoundSummary.round` の 1-indexed 性をコメントで明記
  - `ConvergenceSignal.test.tsx`: チェックマーク glyph が DOM に表示されることを assert する回帰テスト追加
  - `ResultsTab-guards.test.tsx`: `rounds = [round:1, round:2, round:3 with empty expanded_dims]`
    のケースで "Round 3 finished" が表示されることを検証する integration テスト追加
- **影響範囲**:
  - `js/src/components/ConvergenceSignal.tsx`
  - `js/src/tabs/ResultsTab.tsx`
  - `js/src/__tests__/ConvergenceSignal.test.tsx`（1 test 追加）
  - `js/src/__tests__/ResultsTab-guards.test.tsx`（2 tests 追加）
- **互換性**: 純粋な UI 文言修正。traitlet / action / API 変更なし。
- **受け入れ基準**:
  - ユーザー報告の 3 回目 tune で Convergence Signal が「Round 3 finished」と表示される。
  - チェックマーク ✓ が DOM に正しく表示され、`\u2713` の escape 文字列は残らない。
  - 既存 JS 159 + 新規 3 テスト = **162 テスト全パス**。

---

### P-029: Score History Chart を lizyml の Tuning History Plot に一本化

- **日付**: 2026-04-12
- **ステータス**: 承認・実装
- **背景**:
  - P-027 で Widget 独自の `ScoreHistoryChart` コンポーネント（`js/src/components/ScoreHistoryChart.tsx`）を実装したが、
    lizyml 0.9.0 の `Model.tuning_plot()` (`plot_tuning_history`) が同じ機能を backend 側で完結して提供している:
    - Trial 別 scatter（state 色分け）
    - 累積ベストスコアの折れ線
    - ラウンド境界の dashed line（H-0068）
    - 各ラウンドの expanded dims annotation（H-0068）
  - Widget 側の `ScoreHistoryChart` は `tuning_plot` の下位互換であり、**同じ描画を 2 実装持つ重複** となっていた。
    lizyml 側で境界拡張の可視化が拡充されても、Widget 側の自前実装には反映されない。
  - `optimization-history` は既に `available_plots()` に登録されており、PlotViewer で表示可能。
- **提案内容**:
  - **削除**:
    - `js/src/components/ScoreHistoryChart.tsx`
    - `js/src/__tests__/ScoreHistoryChart.test.tsx`
    - `js/src/widget.css` の `.lzw-score-history` 関連スタイル
  - **変更**:
    - `js/src/tabs/ResultsTab.tsx` — Tune 完了時に **Tuning History Accordion を常時表示**し、
      `PlotViewer` に `plotType="optimization-history"` を渡して lizyml backend の Figure を描画する（案B）。
      BoundaryExpansionPanel と ConvergenceSignal の間に配置する。
    - `js/src/tabs/ResultsTab.tsx` — Plots セレクタから `optimization-history` を除外する。
      常時表示 Accordion と重複するため。
  - **保持**:
    - `BoundaryExpansionPanel` — `tuning_plot` に含まれない別情報（per-dim before/after ranges）なので残す。
    - `ConvergenceSignal` — 同上（Fit 画面への遷移導線）。
    - `tune_summary.rounds` / `boundary_report` traitlet — 上記 2 コンポーネントが依然として消費する。
- **影響範囲**:
  - `js/src/components/ScoreHistoryChart.tsx` — **削除**
  - `js/src/__tests__/ScoreHistoryChart.test.tsx` — **削除**
  - `js/src/tabs/ResultsTab.tsx` — imports / レイアウト変更
  - `js/src/widget.css` — スタイル削除
  - `BLUEPRINT.md` §5.x Tune 完了 — 記述更新
  - `HISTORY.md` — 本 proposal
- **互換性**:
  - `tune_summary` traitlet の構造は変更なし（BoundaryExpansionPanel / ConvergenceSignal が使用中）。
  - Adapter Protocol 変更なし。
  - Python API 変更なし。
  - UI 上「Score History」という表示名は「Tuning History」に変わる（ユーザー向けの視認的変更のみ）。
- **代替案（却下）**:
  - **案A: Plots セレクタに含めるだけ** — 1 クリック余分にかかり、Tune 完了時に即座に履歴が見えないため UX が退化。
  - **案C: ScoreHistoryChart を残す** — lizyml 側の機能更新が Widget に反映されない永続的な負債となるため不採用。
- **受け入れ基準**:
  - Tune 完了後、Results タブに "Tuning History" Accordion が常時表示され、
    lizyml backend の `tuning_plot` が描画される。
  - Plots セレクタに `optimization-history` が重複表示されない。
  - `ScoreHistoryChart.tsx` とそのテストが削除される。
  - `.lzw-score-history` CSS 参照が残らない。
  - Python テスト 845 パス（変更なし）、JS vitest は ScoreHistoryChart 関連の 3 tests が削除されて **159 テスト**（162 → 159）が全パス。
  - ruff / mypy / eslint / tsc クリーン。

---

### P-028: Re-tune Launcher（`w.retune()` Python API + `retune` action + UI ボタン）

- **日付**: 2026-04-12
- **ステータス**: 承認・実装
- **背景**:
  - P-027 で Re-tune の **モニタリング** 側（Round progress, Boundary Expansion panel, Score History chart）は実装したが、
    **起動** 側（lizyml 0.9.0 の `Model.tune(resume=True, ...)` を呼び出す経路）は未実装だった。
  - 現状ユーザーは `w.get_model().tune(resume=True, ...)` のように Widget を迂回して直接 backend を呼ぶしかなく、
    その経路では `tune_summary` traitlet が更新されないため Widget UI にラウンド境界や boundary expansion が
    反映されない。
  - A案（起動 + monitoring を end-to-end で繋ぐ）を採用する。
- **提案内容**:
  - **Adapter**:
    - `BackendAdapter.tune()` Protocol に kwargs を追加:
      - `resume: bool = False`
      - `n_trials: int | None = None`
      - `expand_boundary: bool | None = None`
      - `boundary_threshold: float = 0.05`
    - `LizyMLAdapter.tune()` で `model.tune(resume=..., n_trials=..., expand_boundary=..., boundary_threshold=..., progress_callback=...)` に透過。
  - **Service**:
    - `WidgetService.tune()` に同じ kwargs を追加。
    - `resume=True` のとき、`self._model` を再利用する（None の場合は明示的 `ValueError`）。
    - `resume=False` のとき、従来通り `create_model` で新規 model を作る。
  - **Widget Python API**:
    - `LizyWidget.retune(*, n_trials=None, expand_boundary=None, boundary_threshold=0.05, timeout=None) -> LizyWidget`
      を追加。
    - 初回 tune が存在しない場合（`tune_summary` が空）は呼び出し時点で `ValueError`。
  - **Action**:
    - 新 `retune` action を追加（payload: `{n_trials?, expand_boundary?, boundary_threshold?}`）。
    - `_run_job("tune")` 経路を再利用しつつ、`retune_kwargs` を closure で渡す。
    - `job_type` は `"tune"` のまま（Re-tune も Tune の一種として扱う）。
    - traitlet の `progress.round >= 2` で UI 側が Re-tune ラウンドを識別する。
  - **UI (ResultsTab)**:
    - `RetuneControls` を新規コンポーネントとして追加。
    - Best Params Accordion 内の "Apply to Fit" ボタンの下に配置。
    - 入力: `n_trials` (NumericStepper), `expand_boundary` (checkbox), `boundary_threshold` (NumericStepper, 0.0-1.0)。
    - "Re-tune (resume)" ボタン押下で `sendAction("retune", payload)`。
    - 初回 tune 完了後（`hasTune`）のみ表示。
- **影響範囲**:
  - `src/lizyml_widget/adapter.py` — Protocol シグネチャ + LizyMLAdapter 実装（**change gate**）
  - `src/lizyml_widget/service.py` — tune() 拡張、resume 経路
  - `src/lizyml_widget/widget.py` — `retune()` メソッド + `retune` action handler（**change gate**）
  - `js/src/components/RetuneControls.tsx` — 新規
  - `js/src/tabs/ResultsTab.tsx` — RetuneControls 組み込み
  - `js/src/widget.css` — RetuneControls スタイル
  - `tests/test_retune_monitoring.py` — Re-tune 起動経路のテスト追加
  - `js/src/__tests__/RetuneControls.test.tsx` — 新規
  - `BLUEPRINT.md` §3.6 actions、§5.x Tune UI
  - `README.md` — Re-tune 使用例
- **互換性**:
  - Adapter Protocol の新 kwargs は全てデフォルト値付きなので、既存 Adapter 実装（Mock 等）は変更不要。
  - `WidgetService.tune()` / `LizyWidget.tune()` の既存シグネチャは保持（kwargs 追加のみ）。
  - `tune_summary` / `progress` traitlet の構造は P-027 のまま変更なし。
- **代替案（却下）**:
  - **案A: 既存 `tune()` を拡張**（`w.tune(resume=True, n_trials=30)`） — 初回 tune と re-tune の呼び分けが曖昧になり、テスト/ドキュメントが複雑化するため不採用。
  - **案C: 回避策として `w.get_model().tune(...)` を推奨** — Widget UI とモニタリング機能が動作しないため不採用。
- **受け入れ基準**:
  - `w.tune()` → `w.retune(n_trials=30)` で `tune_summary.rounds` が 2 要素になる。
  - `w.retune()` を prior tune なしで呼ぶと明示的 `ValueError`。
  - UI Results タブに "Re-tune (resume)" ボタンが Tune 完了後のみ表示される。
  - UI から Re-tune を発動すると Progress View に Round 2 バッジ、完了後に Score History Chart に
    ラウンド境界が追加される。
  - `resume=True` 時 Service は既存 `self._model` を再利用する。
  - 既存 Python テスト + 新規テスト全パス、既存 vitest + 新規テスト全パス。
  - ruff / mypy / eslint / tsc クリーン。

---

### P-027: Re-tune Monitoring（ラウンド対応 Progress + Boundary Expansion + Score History Chart）

- **日付**: 2026-04-12
- **ステータス**: 承認・実装
- **背景**:
  - LizyML H-0068 で re-tune（Study Resume + Boundary Expansion）が追加され、`TuneProgressInfo` と `TuningResult` に
    新しいフィールドが導入された（GitHub Issue #101）。
    - `TuneProgressInfo.round` / `cumulative_trials` / `expanded_dims`
    - `TuningResult.rounds: tuple[RoundSummary, ...]`
    - `TuningResult.boundary_report: BoundaryReport | None`
  - 既存 Widget の Tune Progress 表示は「単一ラウンド / 単一 trial 番号」が前提で、
    ラウンド境界・累積 trial 数・境界拡張の可視化ができない。
  - Widget の `progress` traitlet は `{current, total, message}` のみで round 情報を持たない。
  - Score History の可視化が未実装（Trial 状態のカウント表示のみ）。
- **提案内容**:
  - **Python**:
    - `types.py`:
      - `TuningSummary` に `rounds: list[dict]` と `boundary_report: dict | None` を追加（必須フィールド、B案）。
      - 既存 `trials` 要素に `round: int` を含める（lizyml 0.9.0 の `TrialResult.round` を透過）。
    - `adapter.py`:
      - `LizyMLAdapter.tune()` の `progress_cb` を拡張し、`round`, `cumulative_trials`, `expanded_dims` を
        `on_progress` payload に含める。
      - `TuningSummary` 生成時に `rounds`, `boundary_report` をシリアライズ。
    - `widget.py`:
      - `progress` traitlet のスキーマを拡張:
        `{current, total, message, round, total_rounds, cumulative_trials, expanded_dims}`（追加は optional な存在）。
      - `tune_summary` に `rounds` / `boundary_report` を格納。
      - `on_progress` シグネチャを後方互換な dict-payload 渡しに拡張
        （`def on_progress(current, total, message, *, round=None, cumulative_trials=None, expanded_dims=None)`）。
    - **バージョン互換**: `lizyml >= 0.9.0` を必須にする（B案）:
      - `pyproject.toml` の `optional-dependencies.lizyml` と `dependency-groups.dev` を `>=0.9.0` に引き上げ。
      - `LizyMLAdapter.__init__` でランタイム version check を実施、`lizyml < 0.9.0` なら明示的 ImportError。
      - README と BLUEPRINT にバージョン互換性マトリクスを追記。
      - `docs/VERSION_COMPAT.md` を新規作成してユーザー向けガイダンスを集約。
  - **TypeScript/JS**:
    - `ProgressView.tsx`:
      - Tune 中に `Round X/Y` バッジと累積 trial 進捗を表示。
      - 前ラウンドのベスト比較（improvement delta）を表示。
    - `BoundaryExpansionPanel.tsx` (新規):
      - `expanded_dims` を受け取り、拡張された dim を方向（lower/upper）と旧値→新値の範囲で表示。
      - 未拡張 dim 数のサマリーを表示。
      - ラウンド 2 以降かつ `expanded_dims` が非空のときのみ表示。
    - `ScoreHistoryChart.tsx` (新規):
      - Plotly.js で trial × score の scatter / line を描画。
      - `rounds` に含まれる境界でラウンド区切り線（vertical dashed）を引く。
      - ラウンドごとに expanded_dims のアノテーションを重ねる。
      - 各 trial の state（COMPLETE/PRUNED/FAIL）で色分け。
      - 既存の Plotly CDN dynamic import パターンを再利用。
    - `ConvergenceSignal.tsx` (新規):
      - 再ラウンドで `expanded_dims` が空のときに "No expansion needed — search space is sufficient" を表示、
        Fit への導線ボタンを提供。
    - `ResultsTab.tsx`:
      - Tune 完了時に Best Params → Boundary Expansion → Score History Chart → Trials Summary の順で表示。
- **影響範囲**:
  - `src/lizyml_widget/types.py` — `TuningSummary` フィールド追加（**change gate**）
  - `src/lizyml_widget/adapter.py` — tune() 進捗払出＆summary 生成
  - `src/lizyml_widget/service.py` — 型中継
  - `src/lizyml_widget/widget.py` — progress traitlet / tune_summary 構造拡張（**change gate**）
  - `js/src/components/ProgressView.tsx` — ラウンド対応
  - `js/src/components/BoundaryExpansionPanel.tsx` — 新規
  - `js/src/components/ScoreHistoryChart.tsx` — 新規
  - `js/src/components/ConvergenceSignal.tsx` — 新規
  - `js/src/tabs/ResultsTab.tsx` — レイアウト拡張
  - `pyproject.toml` — lizyml 制約引き上げ（**change gate**: dependency upper/lower bound 変更）
  - `README.md` — Install/Requirements 節にバージョンマトリクス追記
  - `docs/VERSION_COMPAT.md` — 新規
  - `BLUEPRINT.md` §Tune UI / §Traitlets — 更新
- **互換性（B案: 破壊的変更）**:
  - lizyml < 0.9.0 では動作しない（明示的 ImportError）。
  - ユーザー向けには `pip install "lizyml-widget[lizyml]"` で `lizyml>=0.9.0` を自動解決する仕組みを維持。
  - 既に `lizyml==0.7.x` などの古いバージョンを手動固定しているユーザーへは、
    README の互換性マトリクスで明示し、インストール時に pip resolver が衝突を検出できるよう
    dependency の lower bound を厳密化する。
- **代替案（A案: 後方互換の段階的拡張）**:
  - `progress` traitlet に optional フィールドとして新情報を追加、lizyml 0.7.x でも動く。
  - メリット: 旧版ユーザーを壊さない。
  - デメリット: 新機能（boundary expansion, score history chart）が旧版では常にダミー表示になり、
    JS 分岐が肥大化してメンテナンス負荷が上がる。
  - **採用しない理由**: ユーザー指示により B案を採用。リリースノート・バージョンマトリクスで
    アップグレードガイダンスを丁寧にカバーする方針。
- **受け入れ基準**:
  - `lizyml>=0.9.0` で Tune 実行時に Round/累積 trial 情報が Widget に伝播し、ProgressView に反映される。
  - `expanded_dims` が非空のラウンドで BoundaryExpansionPanel が表示される。
  - Score History Chart が Tune 完了後に表示され、ラウンド境界が可視化される。
  - 再ラウンドで `expanded_dims` が空のとき ConvergenceSignal が表示される。
  - `lizyml<0.9.0` が入っている環境で Widget import 時に明確な ImportError が出る。
  - README / docs にバージョン互換性マトリクスが記載される。
  - 既存 Python テスト + 新規テスト（adapter tune, widget progress, types）が全パス。
  - JS: vitest / biome / tsc が全パス + 新コンポーネントに単体テスト追加。

---

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
