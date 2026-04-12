# LizyML-Widget ↔ LizyML バージョン互換性マトリクス

LizyML-Widget は `lizyml` の ML 仕様（`Config schema`, `TuningResult`,
`BackendAdapter` 契約など）に直接依存するため、
両者のバージョンは **厳密に対応付けて** リリースされる。

> **TL;DR**: 迷ったら `pip install "lizyml-widget[lizyml]"` を使う。
> extras 指定により互換な `lizyml` が自動的に解決される。

---

## 対応表

| lizyml-widget | lizyml               | Python  | 主な新機能 / 破壊的変更                                               |
| ------------- | -------------------- | ------- | --------------------------------------------------------------------- |
| **0.5.x**     | `>=0.9.0, <0.10`     | `>=3.10`| Re-tune monitoring（Round progress, Boundary Expansion, Score History）|
| 0.4.x         | `>=0.7.0, <0.9`      | `>=3.10`| Learning Curve metrics フィルタ, CV strategy metadata                  |
| 0.3.x         | `>=0.5.0, <0.7`      | `>=3.10`| Canonical config, Backend Contract 駆動 UI                             |
| 0.2.x         | `>=0.3.0, <0.5`      | `>=3.10`| Fit/Tune タブ再設計, Apply to Fit                                     |

表の最上段が **最新** 。それ以外のバージョンは過去参照用。

---

## なぜ厳密な範囲指定なのか

1. **契約駆動**: Widget は `lizyml.core.types.tuning_result.TuningResult` や
   `BackendAdapter` Protocol を **直接 import** する。
   フィールド追加・リネームは即座に `AttributeError` / `ImportError` を引き起こす。
2. **Config schema 版管理なし**: LizyML は `schema_version` を段階的にサポートするが、
   Widget が想定する schema 形が異なると UI が空表示になる。
3. **Optuna / LightGBM バージョンの整合性**: `lizyml` が pin している
   `optuna` / `lightgbm` バージョンにより、Widget で受けるオブジェクト（`Trial`,
   `Booster`）の API が変わる。

そのため `lizyml-widget` の optional dependency は:

```toml
[project.optional-dependencies]
lizyml = ["lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10"]
```

のように **lower bound = 0.9.0**, **upper bound = 0.10 未満** の厳密範囲を
指定する。

---

## インストール方法（推奨順）

### 1. Extras を使う（推奨）

```bash
pip install "lizyml-widget[lizyml]"
```

- pip / uv / Poetry の resolver が自動的に互換な `lizyml` を選択する。
- optional extras `[lizyml]` は `lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10`
  を内包する。
- ユーザー側で `lizyml` のバージョンを気にする必要がない。

### 2. 個別指定

すでに `lizyml` を使っているプロジェクトに追加する場合:

```bash
pip install lizyml-widget
# lizyml は既に入っているが、互換バージョンに合わせる必要がある
pip install "lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10"
```

- 既存の `lizyml==0.7.x` などがあると pip resolver が **ResolutionImpossible** を
  検出して警告を出す。
- その場合は `lizyml` を互換バージョンにアップグレードするか、古い Widget を使う
  （対応表を参照）。

### 3. Google Colab / Jupyter

Colab は pip の resolver がラックして古い `lizyml` を強制することがあるため、
以下のセルを使う:

```python
!pip install -q --upgrade "lizyml-widget[lizyml]"
```

- `--upgrade` により既存の `lizyml` も合わせてアップグレードされる。
- 念のため **ランタイム再起動** してから import する。

---

## トラブルシューティング

### `ImportError: lizyml-widget requires lizyml>=0.9.0`

- 旧い `lizyml` がインストールされている。
- 解決策:
  ```bash
  pip install --upgrade "lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10"
  ```

### `AttributeError: 'TuningResult' object has no attribute 'rounds'`

- `lizyml < 0.9.0` が残存している（インストール順の問題）。
- 仮想環境を作り直すか、`pip list | grep lizyml` で実際のバージョンを確認する。

### pip resolver が `ResolutionImpossible` を返す

- 他のパッケージが古い `lizyml` を要求している。
- `pip install --dry-run -vv "lizyml-widget[lizyml]"` で衝突源を特定し、
  該当パッケージを更新するか、Widget 側を古いバージョンに固定する。

---

## ランタイム検証

LizyML-Widget は import 時に `lizyml` の version を検証する。
不一致の場合は **明確なエラーメッセージ** で失敗する:

```python
>>> from lizyml_widget import LizyWidget
>>> w = LizyWidget()
ImportError: lizyml-widget 0.5.0 requires lizyml>=0.9.0,<0.10
  (found: lizyml==0.7.3). Run:
  pip install --upgrade "lizyml[plots,tuning,calibration,explain]>=0.9.0,<0.10"
```

この検証は `LizyMLAdapter.__init__` で行われ、
バックエンドアダプター自体が差し替え可能なため、
**lizyml を使わない別バックエンド**（将来追加予定）では検証されない。

---

## リリース運用

- 新しい `lizyml` のリリースごとに、Widget チームが契約テスト（`tests/test_adapter_*.py`）を
  走らせて互換性を判定する。
- 破壊的変更があれば `lizyml-widget` の minor バージョンを上げ、
  本ドキュメントの対応表を更新する（`HISTORY.md` にも proposal として記録）。
- `HISTORY.md` の対応する `P-` エントリーと相互リンクすること。
