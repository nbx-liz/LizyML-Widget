"""Tests for action dispatch via msg:custom (P-023).

Verifies that _handle_custom_msg correctly routes {type: "action"} messages
to the same action handlers as the traitlet-based dispatch path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo


def _make_widget() -> Any:
    """Create a LizyWidget with mocked LizyML backend."""
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = {"type": "object"}
        adapter.validate_config.return_value = []
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
        adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


class TestActionViaCustomMsg:
    """Actions sent via msg:custom {type: "action"} should invoke handlers."""

    def test_set_target_via_msg(self) -> None:
        """set_target action via msg:custom works like traitlet dispatch."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)

        w._handle_custom_msg(
            {"type": "action", "action_type": "set_target", "payload": {"target": "y"}},
            [],
        )
        assert w.df_info["target"] == "y"

    def test_patch_config_via_msg(self) -> None:
        """patch_config action via msg:custom updates config."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._handle_custom_msg(
            {
                "type": "action",
                "action_type": "patch_config",
                "payload": {
                    "ops": [{"op": "set", "path": "model.params.learning_rate", "value": 0.05}],
                },
            },
            [],
        )
        assert w.config["model"]["params"]["learning_rate"] == 0.05

    def test_set_task_via_msg(self) -> None:
        """set_task action via msg:custom updates df_info."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(100)], "y": range(100)})
        w.load(df, target="y")

        w._handle_custom_msg(
            {"type": "action", "action_type": "set_task", "payload": {"task": "binary"}},
            [],
        )
        assert w.df_info["task"] == "binary"

    def test_unknown_action_via_msg_ignored(self) -> None:
        """Unknown action type via msg:custom should not raise."""
        w = _make_widget()
        # Should not raise
        w._handle_custom_msg(
            {"type": "action", "action_type": "nonexistent", "payload": {}},
            [],
        )

    def test_missing_action_type_ignored(self) -> None:
        """msg:custom action without action_type should not raise."""
        w = _make_widget()
        w._handle_custom_msg({"type": "action", "payload": {}}, [])

    def test_missing_payload_defaults_to_empty(self) -> None:
        """msg:custom action without payload should default to empty dict."""
        w = _make_widget()
        # Should not raise even without payload key
        w._handle_custom_msg({"type": "action", "action_type": "nonexistent"}, [])

    def test_export_yaml_via_msg(self) -> None:
        """export_yaml action via msg:custom sends yaml response."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        w._handle_custom_msg(
            {"type": "action", "action_type": "export_yaml", "payload": {}},
            [],
        )
        # Handler sends {type: "yaml_export", content: ...}
        yaml_msgs = [m for m in sent if m.get("type") == "yaml_export"]
        assert len(yaml_msgs) == 1

    def test_cancel_action_via_msg(self) -> None:
        """cancel action via msg:custom sets cancel flag."""
        w = _make_widget()
        w._handle_custom_msg(
            {"type": "action", "action_type": "cancel", "payload": {}},
            [],
        )
        assert w._cancel_flag.is_set()


class TestActionMsgPayloadValidation:
    """Payload type validation — msg:custom has no traitlet type constraint."""

    def test_non_dict_payload_coerced_to_empty(self) -> None:
        """If payload is not a dict, it should be coerced to {}."""
        w = _make_widget()
        # Should not raise AttributeError
        w._handle_custom_msg(
            {"type": "action", "action_type": "set_target", "payload": "bad"},
            [],
        )

    def test_int_payload_coerced_to_empty(self) -> None:
        """Integer payload should be safely ignored."""
        w = _make_widget()
        w._handle_custom_msg(
            {"type": "action", "action_type": "set_target", "payload": 42},
            [],
        )

    def test_list_payload_coerced_to_empty(self) -> None:
        """List payload should be safely ignored."""
        w = _make_widget()
        w._handle_custom_msg(
            {"type": "action", "action_type": "patch_config", "payload": [1, 2]},
            [],
        )

    def test_null_payload_coerced_to_empty(self) -> None:
        """None payload should be coerced to {}."""
        w = _make_widget()
        w._handle_custom_msg(
            {"type": "action", "action_type": "cancel", "payload": None},
            [],
        )


class TestFitViaMsgCustom:
    """Fit/Tune dispatch via msg:custom reaches _run_job."""

    def test_fit_via_msg_starts_job(self) -> None:
        """fit action via msg:custom should set status to running and start thread."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._handle_custom_msg(
            {"type": "action", "action_type": "fit", "payload": {}},
            [],
        )
        # _run_job sets status before thread.start()
        assert w.status == "running" or w.status == "completed"
        assert w._job_thread is not None

        # Wait for completion
        w._job_thread.join(timeout=30)
        assert w.status in ("completed", "failed")

    def test_tune_via_msg_starts_job(self) -> None:
        """tune action via msg:custom should start job thread."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._handle_custom_msg(
            {"type": "action", "action_type": "tune", "payload": {}},
            [],
        )
        assert w._job_thread is not None
        w._job_thread.join(timeout=60)
        assert w.status in ("completed", "failed")


class TestActionMsgDoesNotInterfereWithPoll:
    """msg:custom action dispatch must not break existing poll handling."""

    def test_poll_still_works_after_action(self) -> None:
        """Poll requests continue to work after action dispatch."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        # Send an action first
        w._handle_custom_msg(
            {"type": "action", "action_type": "cancel", "payload": {}},
            [],
        )

        # Then a poll — should still work
        w._handle_custom_msg({"type": "poll"}, [])
        poll_replies = [m for m in sent if m.get("type") == "job_state"]
        assert len(poll_replies) == 1

    def test_other_msg_types_still_delegate_to_super(self) -> None:
        """Non-action, non-poll messages should still call super()."""
        w = _make_widget()
        # "other" type should not raise and should not be handled as action
        w._handle_custom_msg({"type": "other"}, [])


class TestTraitletActionStillWorks:
    """Python API via traitlet (w.action = {...}) must continue to work."""

    def test_traitlet_action_still_dispatches(self) -> None:
        """Setting action traitlet directly should still trigger _on_action."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)

        w.action = {"type": "set_target", "payload": {"target": "y"}}
        assert w.df_info["target"] == "y"
