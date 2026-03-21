"""Tests for Phase 31A: BlockedGroupKFold CV Strategy backend (Python side).

Test coverage:
- Contract: blocked_group_kfold in cv_strategies
- Service: get_column_stats returns correct distribution
- Service: get_column_stats raises for unknown column
- Service: get_column_stats raises when no data
- Service: preview_splits with expanding mode
- Service: preview_splits with sliding mode
- Service: preview_splits raises when strategy is not blocked_group_kfold
- Service: update_cv with blocked_group_kfold stores blocks/groups
- Service: update_cv with other strategies still works (backward compat)
- Service: build_config for blocked_group_kfold generates nested split
- Widget: get_column_stats action handler
- Widget: preview_splits action handler
- Widget: update_cv with blocked_group_kfold payload
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.adapter_contract import build_capabilities
from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendContract, BackendInfo

# ── Helpers ───────────────────────────────────────────────────


def _mock_adapter() -> Any:
    """Create a mock adapter that delegates config lifecycle to LizyMLAdapter."""
    real_adapter = LizyMLAdapter()
    adapter = MagicMock()
    adapter.info = BackendInfo(name="mock", version="0.0.0")
    adapter.get_config_schema.return_value = {}
    adapter.initialize_config.side_effect = real_adapter.initialize_config
    adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
    adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
    adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
    adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
    adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
    return adapter


def _make_service_with_data(
    n_periods: int = 4,
    rows_per_period: int = 20,
    n_groups: int = 3,
) -> WidgetService:
    """Create a WidgetService with a DataFrame suitable for BlockedGroupKFold."""
    adapter = _mock_adapter()
    svc = WidgetService(adapter=adapter)

    periods = [f"P{i}" for i in range(n_periods)]
    rows_per_combo = rows_per_period // n_groups
    records = []
    for p in periods:
        for g in range(n_groups):
            for _ in range(rows_per_combo):
                records.append({"period": p, "group": g, "x": 1.0, "y": 0})

    df = pd.DataFrame(records)
    svc.load_data(df, target="y")
    return svc


def _make_widget() -> Any:
    """Create a LizyWidget with a fully mocked backend (no lizyml import needed)."""
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.validate_config.return_value = []

        # Provide a minimal config — no lizyml needed
        minimal_config: dict[str, Any] = {
            "config_version": 1,
            "model": {"name": "lgbm", "params": {}},
        }
        adapter.initialize_config.return_value = dict(minimal_config)
        adapter.canonicalize_config.side_effect = lambda cfg, **kw: cfg
        adapter.apply_task_defaults.side_effect = lambda cfg, **kw: cfg
        adapter.apply_config_patch.side_effect = lambda cfg, ops, **kw: cfg
        adapter.prepare_run_config.side_effect = lambda cfg, **kw: cfg
        adapter.classify_best_params.return_value = ({}, {}, {})

        # Return a fully-mocked contract that includes blocked_group_kfold
        mock_contract = BackendContract(
            schema_version=1,
            config_schema={},
            ui_schema={},
            capabilities=build_capabilities(),
        )
        adapter.get_backend_contract.return_value = mock_contract

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


# ── Contract tests ────────────────────────────────────────────


class TestContractBlockedGroupKFold:
    def test_blocked_group_kfold_in_cv_strategies(self) -> None:
        """build_capabilities() must include 'blocked_group_kfold' in cv_strategies."""
        caps = build_capabilities()
        assert "blocked_group_kfold" in caps["cv_strategies"]

    def test_cv_strategies_still_has_existing_entries(self) -> None:
        """Existing cv_strategies must not be removed."""
        caps = build_capabilities()
        for strategy in [
            "kfold",
            "stratified_kfold",
            "time_series",
            "group_time_series",
            "purged_time_series",
            "group_kfold",
            "stratified_group_kfold",
        ]:
            assert strategy in caps["cv_strategies"], f"Missing existing strategy: {strategy}"


# ── Service: get_column_stats ─────────────────────────────────


class TestGetColumnStats:
    def test_returns_correct_distribution(self) -> None:
        """get_column_stats returns value_counts for a categorical column."""
        svc = _make_service_with_data(n_periods=3, rows_per_period=12, n_groups=3)
        result = svc.get_column_stats("period")
        assert result["column"] == "period"
        assert result["unique_count"] == 3
        # Each period should appear rows_per_period times = 12
        values_map = {v["value"]: v["count"] for v in result["values"]}
        assert values_map["P0"] == 12
        assert values_map["P1"] == 12
        assert values_map["P2"] == 12

    def test_returns_dtype_field(self) -> None:
        """get_column_stats includes dtype as string."""
        svc = _make_service_with_data()
        result = svc.get_column_stats("period")
        assert "dtype" in result
        assert isinstance(result["dtype"], str)

    def test_values_sorted_by_index(self) -> None:
        """Values in result are sorted (sort_index on value_counts)."""
        svc = _make_service_with_data(n_periods=4)
        result = svc.get_column_stats("period")
        values = [v["value"] for v in result["values"]]
        assert values == sorted(values)

    def test_raises_for_unknown_column(self) -> None:
        """get_column_stats raises ValueError for unknown column."""
        svc = _make_service_with_data()
        with pytest.raises(ValueError, match="Unknown column"):
            svc.get_column_stats("nonexistent_col")

    def test_raises_when_no_data(self) -> None:
        """get_column_stats raises ValueError when no data is loaded."""
        adapter = _mock_adapter()
        svc = WidgetService(adapter=adapter)
        with pytest.raises(ValueError, match="No data loaded"):
            svc.get_column_stats("any_col")

    def test_int_column_returns_int_counts(self) -> None:
        """Count values are native ints (JSON-serializable)."""
        svc = _make_service_with_data()
        result = svc.get_column_stats("group")
        for entry in result["values"]:
            assert isinstance(entry["count"], int)
            assert isinstance(entry["value"], str)

    def test_unique_count_is_int(self) -> None:
        """unique_count is a native int."""
        svc = _make_service_with_data()
        result = svc.get_column_stats("period")
        assert isinstance(result["unique_count"], int)


# ── Service: preview_splits ───────────────────────────────────


def _set_blocked_cv(
    svc: WidgetService,
    *,
    mode: str = "expanding",
    train_window: int | None = None,
    n_splits: int = 2,
    blocks_col: str = "period",
    groups_col: str = "group",
    cutoffs: list[str] | None = None,
) -> None:
    """Helper: configure blocked_group_kfold CV on service."""
    svc.update_cv(
        "blocked_group_kfold",
        n_splits,
        blocks={
            "col": blocks_col,
            "cutoffs": cutoffs or [],
            "mode": mode,
            "train_window": train_window,
        },
        groups={"col": groups_col, "n_splits": n_splits, "stratify": False, "shuffle": False},
        min_train_rows=1,
        min_valid_rows=1,
    )


class TestPreviewSplitsExpanding:
    def test_time_folds_count(self) -> None:
        """Expanding mode: num_periods - 1 time folds."""
        svc = _make_service_with_data(n_periods=4)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        # 4 periods → 3 time folds
        assert result["time_folds"] == 3

    def test_total_folds_expanding(self) -> None:
        """Total folds = time_folds × group_folds."""
        svc = _make_service_with_data(n_periods=4)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        assert result["total_folds"] == result["time_folds"] * result["group_folds"]
        assert result["group_folds"] == 2

    def test_periods_list_sorted(self) -> None:
        """periods list returned in sorted order."""
        svc = _make_service_with_data(n_periods=4)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        assert result["periods"] == sorted(result["periods"])

    def test_folds_structure(self) -> None:
        """Each fold entry has train_periods, valid_period, and sizes."""
        svc = _make_service_with_data(n_periods=4, rows_per_period=20)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        for fold in result["folds"]:
            assert "train_periods" in fold
            assert "valid_period" in fold
            assert "train_size" in fold
            assert "valid_size" in fold

    def test_expanding_train_periods_grow(self) -> None:
        """In expanding mode, each time fold adds one more train period."""
        svc = _make_service_with_data(n_periods=5)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        # Get unique (fold-level) train_periods by grouping on valid_period
        # Each time fold is represented by n_splits group folds
        seen: dict[str, int] = {}
        for fold in result["folds"]:
            vp = fold["valid_period"]
            train_len = len(fold["train_periods"])
            if vp not in seen:
                seen[vp] = train_len
        train_lengths = list(seen.values())
        # Train periods should grow monotonically
        assert train_lengths == sorted(train_lengths)

    def test_valid_period_is_next_after_train(self) -> None:
        """First time fold: trains on P0, validates on P1."""
        svc = _make_service_with_data(n_periods=4)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        periods = result["periods"]
        # Collect per-valid_period unique train_periods sets
        fold_map: dict[str, list[str]] = {}
        for fold in result["folds"]:
            vp = fold["valid_period"]
            if vp not in fold_map:
                fold_map[vp] = fold["train_periods"]
        # First valid period: P1, train should be [P0]
        first_valid = periods[1]
        assert fold_map[first_valid] == [periods[0]]


class TestPreviewSplitsSliding:
    def test_sliding_time_folds(self) -> None:
        """Sliding mode: same num_periods - 1 time folds as expanding."""
        svc = _make_service_with_data(n_periods=5)
        _set_blocked_cv(svc, mode="sliding", train_window=2, n_splits=2)
        result = svc.preview_splits()
        assert result["time_folds"] == 4  # 5 periods - 1

    def test_sliding_train_window_applied(self) -> None:
        """Sliding mode uses at most train_window periods for training."""
        svc = _make_service_with_data(n_periods=5)
        window = 2
        _set_blocked_cv(svc, mode="sliding", train_window=window, n_splits=2)
        result = svc.preview_splits()
        for fold in result["folds"]:
            assert len(fold["train_periods"]) <= window

    def test_sliding_later_folds_have_correct_window(self) -> None:
        """Later folds in sliding mode use exactly train_window periods."""
        svc = _make_service_with_data(n_periods=6)
        window = 3
        _set_blocked_cv(svc, mode="sliding", train_window=window, n_splits=2)
        result = svc.preview_splits()
        seen_vp: set[str] = set()
        for fold in result["folds"]:
            vp = fold["valid_period"]
            if vp in seen_vp:
                continue
            seen_vp.add(vp)
            # Only check folds that could have a full window
            periods = result["periods"]
            vp_idx = periods.index(vp)
            if vp_idx >= window:
                assert len(fold["train_periods"]) == window


class TestPreviewSplitsGuards:
    def test_raises_when_not_blocked_group_kfold(self) -> None:
        """preview_splits raises ValueError if strategy != 'blocked_group_kfold'."""
        svc = _make_service_with_data()
        svc.update_cv("kfold", 5)
        with pytest.raises(ValueError, match="blocked_group_kfold"):
            svc.preview_splits()

    def test_raises_when_no_data(self) -> None:
        """preview_splits raises ValueError when no data loaded."""
        adapter = _mock_adapter()
        svc = WidgetService(adapter=adapter)
        with pytest.raises(ValueError, match="No data loaded"):
            svc.preview_splits()

    def test_returns_dict_with_required_keys(self) -> None:
        """preview_splits result has all required top-level keys."""
        svc = _make_service_with_data(n_periods=3)
        _set_blocked_cv(svc, mode="expanding", n_splits=2)
        result = svc.preview_splits()
        for key in ("total_folds", "time_folds", "group_folds", "periods", "folds"):
            assert key in result, f"Missing key: {key}"


# ── Service: update_cv with blocked_group_kfold ───────────────


class TestUpdateCvBlockedGroupKFold:
    def test_stores_blocks_config(self) -> None:
        """update_cv with blocked_group_kfold stores blocks nested dict."""
        svc = _make_service_with_data()
        svc.update_cv(
            "blocked_group_kfold",
            3,
            blocks={"col": "period", "cutoffs": [], "mode": "expanding", "train_window": None},
            groups={"col": "group", "n_splits": 3, "stratify": False, "shuffle": False},
            min_train_rows=10,
            min_valid_rows=5,
        )
        cv = svc.get_df_info()["cv"]
        assert cv["strategy"] == "blocked_group_kfold"
        assert cv["blocks"]["col"] == "period"
        assert cv["blocks"]["mode"] == "expanding"
        assert cv["groups"]["col"] == "group"
        assert cv["groups"]["n_splits"] == 3

    def test_stores_min_rows(self) -> None:
        """update_cv stores min_train_rows and min_valid_rows."""
        svc = _make_service_with_data()
        svc.update_cv(
            "blocked_group_kfold",
            2,
            blocks={"col": "period", "cutoffs": [], "mode": "expanding", "train_window": None},
            groups={"col": "group", "n_splits": 2, "stratify": False, "shuffle": False},
            min_train_rows=20,
            min_valid_rows=10,
        )
        cv = svc.get_df_info()["cv"]
        assert cv["min_train_rows"] == 20
        assert cv["min_valid_rows"] == 10

    def test_blocks_none_defaults_stored(self) -> None:
        """update_cv with None blocks stores None for blocks."""
        svc = _make_service_with_data()
        svc.update_cv(
            "blocked_group_kfold",
            2,
            blocks=None,
            groups=None,
            min_train_rows=0,
            min_valid_rows=0,
        )
        cv = svc.get_df_info()["cv"]
        assert cv["strategy"] == "blocked_group_kfold"
        assert cv["blocks"] is None
        assert cv["groups"] is None

    def test_backward_compat_kfold_unchanged(self) -> None:
        """update_cv with kfold still works as before (backward compat)."""
        svc = _make_service_with_data()
        svc.update_cv("kfold", 5)
        cv = svc.get_df_info()["cv"]
        assert cv["strategy"] == "kfold"
        assert cv["n_splits"] == 5
        # blocks/groups keys should not be present for kfold
        assert "blocks" not in cv or cv.get("blocks") is None

    def test_backward_compat_group_kfold_unchanged(self) -> None:
        """update_cv with group_kfold and group_column still works."""
        svc = _make_service_with_data()
        svc.update_cv("group_kfold", 4, group_column="group")
        cv = svc.get_df_info()["cv"]
        assert cv["strategy"] == "group_kfold"
        assert cv["group_column"] == "group"


# ── Service: build_config for blocked_group_kfold ─────────────


_MINIMAL_USER_CONFIG: dict[str, Any] = {
    "config_version": 1,
    "model": {"name": "lgbm", "params": {}},
}


def _mock_adapter_with_minimal_init() -> Any:
    """Create a mock adapter that returns a minimal config (no lizyml import needed)."""
    adapter = MagicMock()
    adapter.info = BackendInfo(name="mock", version="0.0.0")
    adapter.initialize_config.return_value = dict(_MINIMAL_USER_CONFIG)
    adapter.prepare_run_config.return_value = dict(_MINIMAL_USER_CONFIG)
    adapter.canonicalize_config.side_effect = lambda cfg, **kw: cfg
    adapter.apply_task_defaults.side_effect = lambda cfg, **kw: cfg
    return adapter


class TestBuildConfigBlockedGroupKFold:
    def _make_svc(self) -> WidgetService:
        adapter = _mock_adapter_with_minimal_init()
        svc = WidgetService(adapter=adapter)
        # Build a minimal df with period/group cols
        records = []
        for p in range(4):
            for g in range(3):
                for _ in range(5):
                    records.append({"period": f"P{p}", "group": g, "x": 1.0, "y": 0})
        df = pd.DataFrame(records)
        svc.load_data(df, target="y")
        return svc

    def test_generates_nested_split_section(self) -> None:
        """build_config for blocked_group_kfold emits nested blocks/groups in split."""
        svc = self._make_svc()
        svc.update_cv(
            "blocked_group_kfold",
            3,
            blocks={"col": "period", "cutoffs": [], "mode": "expanding", "train_window": None},
            groups={"col": "group", "n_splits": 3, "stratify": False, "shuffle": False},
            min_train_rows=10,
            min_valid_rows=5,
        )
        result = svc.build_config(dict(_MINIMAL_USER_CONFIG))
        split = result["split"]
        assert split["method"] == "blocked_group_kfold"
        assert "blocks" in split
        assert "groups" in split
        assert split["blocks"]["col"] == "period"
        assert split["blocks"]["mode"] == "expanding"
        assert split["groups"]["col"] == "group"
        assert split["groups"]["n_splits"] == 3

    def test_nested_split_has_min_rows(self) -> None:
        """build_config includes min_train_rows and min_valid_rows in split."""
        svc = self._make_svc()
        svc.update_cv(
            "blocked_group_kfold",
            2,
            blocks={"col": "period", "cutoffs": [], "mode": "expanding", "train_window": None},
            groups={"col": "group", "n_splits": 2, "stratify": False, "shuffle": False},
            min_train_rows=20,
            min_valid_rows=10,
        )
        result = svc.build_config(dict(_MINIMAL_USER_CONFIG))
        split = result["split"]
        assert split["min_train_rows"] == 20
        assert split["min_valid_rows"] == 10

    def test_no_n_splits_top_level_for_blocked(self) -> None:
        """For blocked_group_kfold, n_splits at top-level split is NOT emitted
        (it lives inside groups.n_splits)."""
        svc = self._make_svc()
        svc.update_cv(
            "blocked_group_kfold",
            3,
            blocks={"col": "period", "cutoffs": [], "mode": "expanding", "train_window": None},
            groups={"col": "group", "n_splits": 3, "stratify": False, "shuffle": False},
            min_train_rows=0,
            min_valid_rows=0,
        )
        result = svc.build_config(dict(_MINIMAL_USER_CONFIG))
        split = result["split"]
        # n_splits should NOT be at top-level of split for blocked_group_kfold
        assert "n_splits" not in split

    def test_other_strategies_still_emit_n_splits(self) -> None:
        """build_config for kfold still emits n_splits at top-level (backward compat)."""
        svc = self._make_svc()
        svc.update_cv("kfold", 5)
        result = svc.build_config(dict(_MINIMAL_USER_CONFIG))
        split = result["split"]
        assert split["n_splits"] == 5

    def test_sliding_mode_with_train_window(self) -> None:
        """Sliding mode includes train_window in blocks section."""
        svc = self._make_svc()
        svc.update_cv(
            "blocked_group_kfold",
            2,
            blocks={"col": "period", "cutoffs": [], "mode": "sliding", "train_window": 3},
            groups={"col": "group", "n_splits": 2, "stratify": False, "shuffle": False},
            min_train_rows=0,
            min_valid_rows=0,
        )
        result = svc.build_config(dict(_MINIMAL_USER_CONFIG))
        split = result["split"]
        assert split["blocks"]["mode"] == "sliding"
        assert split["blocks"]["train_window"] == 3


# ── Widget action handler tests ───────────────────────────────


class TestWidgetGetColumnStats:
    def test_sends_column_stats_response(self) -> None:
        """_handle_get_column_stats sends column stats via custom message."""
        w = _make_widget()
        n_periods = 4
        rows_per_period = 20
        records = []
        for p in range(n_periods):
            for _ in range(rows_per_period):
                records.append({"period": f"P{p}", "x": 1.0, "y": 0})
        df = pd.DataFrame(records)
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "get_column_stats", "payload": {"column": "period"}}

        assert len(sent) == 1
        msg = sent[0]
        assert msg["type"] == "column_stats"
        assert msg["column"] == "period"
        assert msg["unique_count"] == n_periods
        assert len(msg["values"]) == n_periods

    def test_missing_column_sends_error(self) -> None:
        """_handle_get_column_stats sends error message for unknown column."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "get_column_stats", "payload": {"column": "nonexistent"}}

        assert len(sent) == 1
        assert sent[0]["type"] == "column_stats_error"

    def test_missing_column_key_sends_error(self) -> None:
        """_handle_get_column_stats with empty column name sends error."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "get_column_stats", "payload": {}}

        assert len(sent) == 1
        assert sent[0]["type"] == "column_stats_error"


class TestWidgetPreviewSplits:
    def test_sends_preview_splits_response(self) -> None:
        """_handle_preview_splits sends preview_splits response via custom message."""
        w = _make_widget()
        n_periods = 4
        records = []
        for p in range(n_periods):
            for g in range(3):
                for _ in range(5):
                    records.append({"period": f"P{p}", "group": g, "x": 1.0, "y": 0})
        df = pd.DataFrame(records)
        w.load(df, target="y")

        # Set blocked_group_kfold strategy
        w.action = {
            "type": "update_cv",
            "payload": {
                "strategy": "blocked_group_kfold",
                "n_splits": 2,
                "blocks": {
                    "col": "period",
                    "cutoffs": [],
                    "mode": "expanding",
                    "train_window": None,
                },
                "groups": {
                    "col": "group",
                    "n_splits": 2,
                    "stratify": False,
                    "shuffle": False,
                },
                "min_train_rows": 1,
                "min_valid_rows": 1,
            },
        }

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "preview_splits", "payload": {}}

        assert len(sent) == 1
        msg = sent[0]
        assert msg["type"] == "preview_splits"
        assert "total_folds" in msg
        assert "time_folds" in msg
        assert "periods" in msg

    def test_preview_splits_error_sends_error_msg(self) -> None:
        """_handle_preview_splits sends error message when strategy is wrong."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
        w.load(df, target="y")
        # Strategy is kfold by default, not blocked_group_kfold
        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "preview_splits", "payload": {}}

        assert len(sent) == 1
        assert sent[0]["type"] == "preview_splits_error"


class TestWidgetUpdateCvBlockedGroupKFold:
    def test_update_cv_with_blocked_group_kfold_payload(self) -> None:
        """_handle_update_cv passes blocks/groups to service for blocked_group_kfold."""
        w = _make_widget()
        records = []
        for p in range(4):
            for g in range(3):
                for _ in range(5):
                    records.append({"period": f"P{p}", "group": g, "x": 1.0, "y": 0})
        df = pd.DataFrame(records)
        w.load(df, target="y")

        w.action = {
            "type": "update_cv",
            "payload": {
                "strategy": "blocked_group_kfold",
                "n_splits": 2,
                "blocks": {
                    "col": "period",
                    "cutoffs": [],
                    "mode": "expanding",
                    "train_window": None,
                },
                "groups": {
                    "col": "group",
                    "n_splits": 2,
                    "stratify": False,
                    "shuffle": False,
                },
                "min_train_rows": 5,
                "min_valid_rows": 3,
            },
        }

        cv = w.df_info["cv"]
        assert cv["strategy"] == "blocked_group_kfold"
        assert cv["blocks"]["col"] == "period"
        assert cv["groups"]["col"] == "group"
        assert cv["min_train_rows"] == 5
        assert cv["min_valid_rows"] == 3

    def test_update_cv_blocked_invalid_strategy_rejected(self) -> None:
        """_handle_update_cv still rejects invalid strategy even with nested blocks."""
        w = _make_widget()
        # Load backend_contract to get valid strategies (includes blocked_group_kfold)
        df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
        w.load(df, target="y")

        w.action = {
            "type": "update_cv",
            "payload": {
                "strategy": "totally_invalid",
                "n_splits": 2,
                "blocks": {"col": "period"},
            },
        }
        assert w.error.get("code") == "CV_ERROR"
