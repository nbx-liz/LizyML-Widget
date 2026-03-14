# LizyML 仕様変更依頼: multiclassova 使用時の確率正規化

## ステータス

- 起票日: 2026-03-14
- 対象: LizyML 0.1.2
- 起票元: LizyML-Widget (multiclass Fit で AUC 評価エラー)

## 1. 問題

`objective="multiclassova"` で学習した場合、evaluation metrics に `auc` を含めると
sklearn の `roc_auc_score` が以下のエラーを発生させる。

```
ValueError: Target scores need to be probabilities for multiclass roc_auc,
            i.e. they should sum up to 1.0 over classes
```

### 再現条件

- task = `"multiclass"`
- model.params.objective = `"multiclassova"` (または `"softmax"`)
- evaluation.metrics に `"auc"` を含む

### 根本原因

LightGBM の `multiclassova` objective は各クラスに独立した sigmoid を適用するため、
`booster.predict(X)` の出力が行ごとに合計 1.0 にならない。

```
multiclass  (softmax) : predict() → [0.17, 0.003, 0.82]  合計 ≈ 1.0 ✓
multiclassova (OVA)   : predict() → [0.34, 0.007, 0.96]  合計 ≈ 1.31 ✗
```

sklearn の `roc_auc_score(multi_class="ovr")` は合計 1.0 をハードバリデーション
しており、非正規化の出力を渡すと ValueError になる。

### 影響メトリクス

`needs_proba=True` のメトリクスのうち、確率分布を前提とするもの:

| メトリクス | 影響 | 理由 |
|-----------|------|------|
| `auc` | **エラー** | `roc_auc_score(multi_class="ovr")` が合計 1.0 を検証 |
| `brier` | 値が不正確 | `brier_score_loss` は [0,1] 範囲の確率を前提 |
| `logloss` | 値が不正確 | `log_loss` は確率分布を前提 |
| `auc_pr` | 影響なし | クラス別独立計算のため正規化不要 |

## 2. 提案: Evaluator 層での行正規化 (案 B)

### 設計方針

- `predict_proba()` の契約は変更しない（生の sigmoid 出力を返し続ける）
- 評価パイプラインの責務として、メトリクス計算前に正規化する
- 変更箇所は `evaluation/evaluator.py` の `_pred_for_metric()` 1 関数のみ

### 変更対象

**ファイル**: `lizyml/evaluation/evaluator.py`
**関数**: `_pred_for_metric()` (L18-35)

### 現行コード

```python
def _pred_for_metric(
    metric: BaseMetric,
    raw_pred: npt.NDArray[np.float64],
    task: TaskType,
) -> npt.NDArray[Any]:
    """Return the appropriate prediction array for *metric*.

    - ``needs_proba=True``: return ``raw_pred`` as-is.
    - ``needs_proba=False``: binarise (binary) or argmax (multiclass) as needed.
    """
    if metric.needs_proba:
        return raw_pred                          # ← ここが問題
    if task == "binary":
        return (raw_pred >= 0.5).astype(int)
    if task == "multiclass":
        result: npt.NDArray[np.intp] = raw_pred.argmax(axis=1)
        return result
    return raw_pred  # regression
```

### 修正案

```python
def _normalize_multiclass_proba(
    pred: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Row-wise normalize multiclass predictions to sum to 1.0.

    Required when the objective is ``multiclassova`` (independent sigmoid
    per class).  For ``multiclass`` (softmax), predictions already sum to
    1.0 and this operation is idempotent.
    """
    row_sums = pred.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (degenerate edge case)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return pred / row_sums


def _pred_for_metric(
    metric: BaseMetric,
    raw_pred: npt.NDArray[np.float64],
    task: TaskType,
) -> npt.NDArray[Any]:
    """Return the appropriate prediction array for *metric*.

    - ``needs_proba=True``:
      - multiclass 2-D predictions are row-normalised so that metrics
        receiving probabilities always see a valid distribution (sum = 1).
        This is necessary for ``multiclassova`` (independent sigmoid),
        and idempotent for ``multiclass`` (softmax).
      - Other tasks: return ``raw_pred`` as-is.
    - ``needs_proba=False``: binarise (binary) or argmax (multiclass).
    """
    if metric.needs_proba:
        if task == "multiclass" and raw_pred.ndim == 2:
            return _normalize_multiclass_proba(raw_pred)
        return raw_pred
    if task == "binary":
        return (raw_pred >= 0.5).astype(int)
    if task == "multiclass":
        result: npt.NDArray[np.intp] = raw_pred.argmax(axis=1)
        return result
    return raw_pred  # regression
```

## 3. 変更の性質

### 変更しないもの

| コンポーネント | 理由 |
|---------------|------|
| `predict_proba()` | 生の sigmoid 出力を保持する要件 |
| `predict_raw()` | logits アクセスは変更不要 |
| `predict()` | argmax → クラスラベル、入力の正規化に依存しない |
| `BaseMetric` Protocol | 属性追加なし |
| `_TASK_METRICS` (registry) | `"auc"` を multiclass から除外しない |
| 個別メトリクスクラス | AUC, Brier, LogLoss 等の __call__ は変更不要 |

### 冪等性

`multiclass` (softmax) の場合、出力は既に合計 1.0 なので正規化は実質 no-op。
追加のオーバーヘッドは `sum()` + 比較のみで無視できる。

### argmax への影響

`needs_proba=False` のメトリクス（`accuracy`, `f1`）は `argmax` を使う。
行正規化は各行の相対順序を保存するため `argmax` の結果は変わらない。
ただし `needs_proba=False` のパスでは正規化を行わないので、そもそも影響しない。

## 4. テスト方針

### 追加すべきテストケース

```python
class TestMulticlassOvaNormalization:
    """_pred_for_metric normalizes multiclassova predictions."""

    def test_softmax_pred_unchanged(self):
        """Softmax predictions (already sum=1) are returned as-is."""
        pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
        result = _pred_for_metric(auc_metric, pred, "multiclass")
        np.testing.assert_allclose(result, pred)

    def test_ova_sigmoid_pred_normalized(self):
        """OVA sigmoid predictions are row-normalized to sum=1."""
        pred = np.array([[0.34, 0.007, 0.96], [0.5, 0.5, 0.8]])
        result = _pred_for_metric(auc_metric, pred, "multiclass")
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        # Relative order preserved
        assert result[0, 2] > result[0, 0] > result[0, 1]

    def test_zero_row_handled(self):
        """All-zero row does not cause division by zero."""
        pred = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.4]])
        result = _pred_for_metric(auc_metric, pred, "multiclass")
        assert np.all(np.isfinite(result))

    def test_binary_not_affected(self):
        """Binary predictions are not row-normalized."""
        pred = np.array([0.2, 0.8, 0.5])
        result = _pred_for_metric(auc_metric, pred, "binary")
        np.testing.assert_array_equal(result, pred)

    def test_needs_proba_false_not_affected(self):
        """Metrics with needs_proba=False skip normalization."""
        pred = np.array([[0.34, 0.007, 0.96]])
        result = _pred_for_metric(accuracy_metric, pred, "multiclass")
        assert result.ndim == 1  # argmax applied, not normalized
```

### 統合テスト

```python
def test_auc_with_multiclassova_predictions():
    """AUC metric succeeds with non-normalized multiclassova predictions."""
    y_true = np.array([0, 1, 2, 0, 1])
    # Simulated multiclassova output (sums > 1.0)
    y_pred = np.array([
        [0.8, 0.1, 0.3],
        [0.2, 0.9, 0.1],
        [0.1, 0.2, 0.95],
        [0.7, 0.3, 0.2],
        [0.1, 0.85, 0.15],
    ])
    assert y_pred.sum(axis=1).max() > 1.0  # Confirm not normalized

    normalized = _normalize_multiclass_proba(y_pred)
    score = roc_auc_score(y_true, normalized, multi_class="ovr", average="macro")
    assert 0.0 <= score <= 1.0
```

## 5. データフロー図 (修正後)

```
predict_proba()
  │  (n, k) sigmoid 出力 — 生値のまま
  ▼
get_fold_pred()
  │  そのまま通過 (変更なし)
  ▼
_compute_metrics()
  │
  ├─ _pred_for_metric(needs_proba=True, multiclass, 2D)
  │    └─ _normalize_multiclass_proba()   ← 追加
  │         行正規化 → 合計 1.0
  │         ▼
  │    AUC / Brier / LogLoss に正規化済み確率を渡す
  │
  └─ _pred_for_metric(needs_proba=False, multiclass, 2D)
       └─ argmax → クラスラベル (変更なし)
            ▼
       Accuracy / F1 にラベルを渡す
```

## 6. 参考情報

- [LightGBM: multiclass vs multiclassova](https://github.com/Microsoft/LightGBM/issues/1518)
- [sklearn roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) — `multi_class="ovr"` は確率分布を要求
- [LightGBM Parameters: objective](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) — `multiclassova` は独立 sigmoid
