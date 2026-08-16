from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.tool_executor import (
    _cache_post_tool_context_metadata,
    _finalize_keyboard_interrupt_batch,
    _finalize_tool_result_batch,
)
from hermes_state import SessionDB
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import enforce_turn_budget


def _tool_call(name: str, call_id: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_post_tool_context_runs_after_budget_and_persists_exact_provider_bytes(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "post-tool-context"
    db.create_session(session_id, source="cli")
    first_original = "x" * 80_000
    second_original = "y" * 80_000
    db.append_message(
        session_id,
        "tool",
        content=first_original,
        tool_name="read_file",
        tool_call_id="call-1",
    )
    db.append_message(
        session_id,
        "tool",
        content=second_original,
        tool_name="read_file",
        tool_call_id="call-2",
    )
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=db,
        _current_turn_id="turn-1",
        platform="cli",
        model="test-model",
    )
    tool_messages = [
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call-1",
            "content": first_original,
        },
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call-2",
            "content": second_original,
        },
    ]
    tool_calls = [
        _tool_call("read_file", "call-1", '{"path":"one"}'),
        _tool_call("read_file", "call-2", '{"path":"two"}'),
    ]

    def _hook(name, **kwargs):
        assert name == "post_tool_context"
        if kwargs["tool_call_id"] == "call-1":
            return [{"context": "SAME-TURN-DELTA"}]
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "post_tool_context"),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_hook),
    ):
        enforce_turn_budget(
            tool_messages,
            env=None,
            config=BudgetConfig(turn_budget=100_000, preview_size=1_000),
        )
        tool_messages[0]["content"] += "\n\n[STEER] latest user guidance"
        changed = _finalize_tool_result_batch(
            agent,
            tool_messages,
            tool_calls,
            effective_task_id="task-1",
            api_call_count=2,
        )

    assert changed is True
    assert len(tool_messages[0]["content"]) < len(first_original)
    assert tool_messages[0]["content"].endswith("\n\nSAME-TURN-DELTA")
    assert "[STEER] latest user guidance" in tool_messages[0]["content"]
    replay = db.get_messages_as_conversation(session_id)
    assert replay[0]["content"] == tool_messages[0]["content"]
    assert replay[1]["content"] == tool_messages[1]["content"]
    db.close()


def test_post_tool_context_appends_to_multimodal_result(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "multimodal-context"
    db.create_session(session_id, source="cli")
    original = [
        {"type": "text", "text": "screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    db.append_message(
        session_id,
        "tool",
        content=original,
        tool_name="computer_use",
        tool_call_id="call-image",
    )
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=db,
        _current_turn_id="turn-image",
    )
    messages = [
        {
            "role": "tool",
            "name": "computer_use",
            "tool_call_id": "call-image",
            "content": original,
        }
    ]

    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "IMAGE-DELTA"}],
        ),
    ):
        _finalize_tool_result_batch(
            agent,
            messages,
            [_tool_call("computer_use", "call-image")],
            effective_task_id="task-image",
            api_call_count=1,
        )

    assert messages[0]["content"][-1] == {
        "type": "text",
        "text": "\n\nIMAGE-DELTA",
    }
    replay = db.get_messages_as_conversation(session_id)
    assert replay[0]["content"] == messages[0]["content"]
    db.close()


def test_session_db_tool_result_update_is_scoped_by_call_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "tool", content="old-1", tool_call_id="call-1")
    db.append_message("s1", "tool", content="old-2", tool_call_id="call-2")

    assert db.set_tool_result_content("s1", "call-1", "new-1") == 1
    rows = db.get_messages_as_conversation("s1")
    assert rows[0]["content"] == "new-1"
    assert rows[1]["content"] == "old-2"
    db.close()


def test_post_tool_context_uses_effective_execution_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("effective", source="cli")
    db.append_message(
        "effective", "tool", content="result", tool_call_id="call-1"
    )
    agent = SimpleNamespace(
        session_id="effective",
        _session_db=db,
        _current_turn_id="turn-1",
    )
    _cache_post_tool_context_metadata(
        agent,
        tool_call_id="call-1",
        function_name="write_file",
        function_args={"path": "/effective/path"},
    )
    message = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result",
    }
    seen = {}

    def _hook(_name, **kwargs):
        seen.update(kwargs)
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_hook),
    ):
        _finalize_tool_result_batch(
            agent,
            [message],
            [_tool_call("tool_call", "call-1", '{"path":"stale"}')],
            effective_task_id="task",
            api_call_count=1,
        )

    assert seen["tool_name"] == "write_file"
    assert seen["args"] == {"path": "/effective/path"}
    assert agent._post_tool_context_metadata == {}
    db.close()


def test_final_tool_result_update_failure_keeps_reason_contract():
    class LockedDB:
        def set_tool_result_content(self, *_args):
            raise RuntimeError("database is locked")

    agent = SimpleNamespace(
        session_id="locked",
        _session_db=LockedDB(),
        _current_turn_id="turn-1",
        _incremental_persistence_failed=False,
    )
    with patch("hermes_cli.lifecycle.has_hook", return_value=False):
        _finalize_tool_result_batch(
            agent,
            [{"role": "tool", "tool_call_id": "call-1", "content": "result"}],
            [_tool_call("read_file", "call-1")],
            effective_task_id="task",
            api_call_count=1,
        )

    assert agent._incremental_persistence_failed is True
    assert agent._last_persistence_error_cause == "locked"


def test_keyboard_interrupt_batch_reaches_post_tool_context(monkeypatch):
    seen = []
    agent = SimpleNamespace(
        session_id="cancelled",
        _session_db=None,
        _current_turn_id="turn-1",
        _apply_internal_events_to_tool_results_and_flush=lambda *args, **kwargs: None,
        _apply_pending_steer_to_tool_results=lambda *args, **kwargs: None,
    )
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="terminal", arguments='{"command":"sleep 1"}'),
    )
    messages = []

    def invoke(name, **kwargs):
        seen.append((name, kwargs["tool_call_id"]))
        return [{"context": "CANCEL-CONTEXT"}]

    monkeypatch.setattr(
        "agent.tool_executor._flush_session_db_after_tool_progress",
        lambda *args, **kwargs: True,
    )
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), patch(
        "hermes_cli.lifecycle.invoke_hook", side_effect=invoke
    ):
        _finalize_keyboard_interrupt_batch(
            agent,
            messages,
            [call],
            current_function_name="terminal",
            current_function_args={"command": "sleep 1"},
            effective_task_id="",
            api_call_count=2,
        )

    assert seen == [("post_tool_context", "call-1")]
    assert messages[0]["tool_call_id"] == "call-1"
    assert messages[0]["content"].endswith("CANCEL-CONTEXT")
