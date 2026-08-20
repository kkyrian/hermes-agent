from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from agent.tool_executor import (
    _cache_post_tool_context_metadata,
    _finalize_keyboard_interrupt_batch,
    _finalize_tool_result_batch,
    _finalize_tool_boundary,
)
from hermes_state import SessionDB
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import enforce_turn_budget


def _tool_call(name: str, call_id: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _append_internal(messages, _count):
    messages[-1]["content"] += "\n\nINTERNAL-EVIDENCE"
    return True


def _append_steer(messages, _count):
    messages[-1]["content"] += "\n\nSTEER-GUIDANCE"


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


def test_post_tool_context_is_rebounded_before_final_persistence(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "post-hook-rebudget"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "tool", content="x" * 40_000, tool_call_id="call-1")
    db.append_message(session_id, "tool", content="y" * 40_000, tool_call_id="call-2")
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=db,
        _current_turn_id="turn-rebudget",
        _post_tool_context_metadata={},
        _incremental_persistence_failed=False,
        _flush_messages_to_session_db=lambda *_args, **_kwargs: True,
        _apply_pending_internal_events_to_tool_results=_append_internal,
        _apply_pending_steer_to_tool_results=_append_steer,
    )
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 40_000},
        {"role": "tool", "tool_call_id": "call-2", "content": "y" * 40_000},
    ]
    calls = [_tool_call("read_file", "call-1"), _tool_call("read_file", "call-2")]

    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "HOOK-CONTEXT-" + "z" * 40_000}],
        ),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            calls,
            effective_task_id="task-rebudget",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=50_000, preview_size=1_000),
        )

    assert sum(len(message["content"]) for message in messages) < 60_000
    assert messages[-1]["content"].index("INTERNAL-EVIDENCE") < messages[-1][
        "content"
    ].index("STEER-GUIDANCE") < messages[-1]["content"].index("HOOK-CONTEXT")
    replay = db.get_messages_as_conversation(session_id)
    assert [row["content"] for row in replay] == [
        message["content"] for message in messages
    ]
    db.close()


def test_hook_suffixes_spill_or_truncate_before_small_turn_budget(tmp_path):
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-small-budget",
        _apply_pending_internal_events_to_tool_results=_append_internal,
        _apply_pending_steer_to_tool_results=_append_steer,
    )
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "base"},
    ]
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "HOOK-" + "z" * 20_000}],
        ),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("terminal", "call-1")],
            effective_task_id="task-small-budget",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=120, preview_size=40),
        )

    content = messages[0]["content"]
    assert "INTERNAL-EVIDENCE" in content
    assert "STEER-GUIDANCE" in content
    assert content.index("INTERNAL-EVIDENCE") < content.index("STEER-GUIDANCE")
    assert "z" * 1_000 not in content
    assert "Post-tool context" in content
    assert len(content) < 500


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


def test_post_tool_context_spills_joined_aggregate_and_persists_preview(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("aggregate", source="cli")
    db.append_message("aggregate", "tool", content="result", tool_call_id="call-1")
    agent = SimpleNamespace(
        session_id="aggregate",
        _session_db=db,
        _current_turn_id="turn-1",
    )
    message = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    cfg = {
        "enabled": True,
        "max_chars": 100,
        "preview_head": 20,
        "preview_tail": 20,
        "directory": str(tmp_path / "spills"),
    }
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "A" * 70}, {"context": "B" * 70}],
        ),
        patch("tools.hook_output_spill.get_spill_config", return_value=cfg),
    ):
        _finalize_tool_result_batch(
            agent,
            [message],
            [_tool_call("read_file", "call-1")],
            effective_task_id="task",
            api_call_count=1,
        )

    assert "post_tool_context output truncated" in message["content"]
    saved_path = Path(message["content"].split("full content saved to ", 1)[1].split("]", 1)[0])
    assert saved_path.read_text(encoding="utf-8") == "A" * 70 + "\n\n" + "B" * 70
    assert db.get_messages_as_conversation("aggregate")[0]["content"] == message["content"]
    db.close()


def test_mandatory_spill_failure_persists_complete_post_tool_context(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("spill-failure", source="cli")
    db.append_message("spill-failure", "tool", content="result", tool_call_id="call-1")
    agent = SimpleNamespace(
        session_id="spill-failure",
        _session_db=db,
        _current_turn_id="turn-1",
    )
    message = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    context = "HEAD-" + "x" * 500 + "-UNIQUE-MIDDLE-" + "y" * 500 + "-TAIL"
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[
                {
                    "context": context,
                    "spill_required": True,
                    "spill_failure_action": "Spill failed.",
                }
            ],
        ),
        patch(
            "tools.hook_output_spill.write_spill_file",
            return_value={"ok": False, "path": None, "error": "ENOSPC"},
        ),
    ):
        _finalize_tool_result_batch(
            agent,
            [message],
            [_tool_call("read_file", "call-1")],
            effective_task_id="task",
            api_call_count=1,
        )

    assert "mandatory spill failed" in message["content"]
    assert context in message["content"]
    assert "UNIQUE-MIDDLE" in message["content"]
    assert db.get_messages_as_conversation("spill-failure")[0]["content"] == message["content"]
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
