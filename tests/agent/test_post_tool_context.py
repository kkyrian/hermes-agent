from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

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


def test_post_tool_context_receives_authority_rebudgeted_result():
    seen_results = []
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-final-hook-input",
        _apply_pending_internal_events_to_tool_results=lambda *_args: None,
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 5_000},
        {"role": "tool", "tool_call_id": "call-2", "content": "small"},
    ]

    def hook(_name, **kwargs):
        seen_results.append(kwargs["result"])
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=hook),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("terminal", "call-1"), _tool_call("terminal", "call-2")],
            effective_task_id="task-final-hook-input",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=500, preview_size=80),
        )

    assert seen_results == [message["content"] for message in messages]
    assert "x" * 1_000 not in seen_results[0]


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
    assert len(content) <= 120


def test_evidence_in_one_result_does_not_exempt_other_results_from_rebudgeting():
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-mixed-evidence",
        _apply_pending_internal_events_to_tool_results=lambda messages, _count: (
            messages[-1].__setitem__("content", messages[-1]["content"] + "\n\n" + "A" * 180)
        ),
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    evidence = "<persisted-output>durable path</persisted-output>"
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": evidence,
            "_trusted_budget_evidence": True,
        },
        {"role": "tool", "tool_call_id": "call-2", "content": "x" * 140},
    ]
    _finalize_tool_boundary(
        agent,
        messages,
        messages,
        [_tool_call("terminal", "call-1"), _tool_call("terminal", "call-2")],
        effective_task_id="task-mixed-evidence",
        api_call_count=1,
        budget=BudgetConfig(turn_budget=220, preview_size=40),
    )

    assert messages[0]["content"] == evidence
    assert "x" * 100 not in messages[1]["content"]
    assert messages[1]["content"].endswith("A" * 180)


def test_internal_completion_evidence_is_persisted_before_authority_rebudget():
    agent = SimpleNamespace(
        session_id="session-completions",
        _current_turn_id="turn-completions",
        _session_db=None,
        _apply_pending_internal_events_to_tool_results=lambda messages, _count: (
            messages[-1].__setitem__(
                "content", messages[-1]["content"] + "\n\n" + "E" * 5_000
            )
        ),
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "base"}]
    persisted = "<persisted-output>\nFull output saved to: /tmp/completions.txt\n</persisted-output>"

    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        return_value=persisted,
    ) as persist:
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("terminal", "call-1")],
            effective_task_id="task-completions",
            api_call_count=3,
            budget=BudgetConfig(turn_budget=1_000, preview_size=100),
        )

    assert messages[0]["content"].endswith(persisted)
    assert "E" * 1_000 not in messages[0]["content"]
    assert len(messages[0]["content"]) <= 1_000
    assert persist.call_args.kwargs["threshold"] == 1_000
    assert persist.call_args.args[0].endswith("E" * 5_000)


def test_mandatory_spill_failure_survives_final_small_budget():
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-mandatory-failure-budget",
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    context = "HEAD-" + "x" * 500 + "-UNIQUE-MIDDLE-" + "y" * 500 + "-TAIL"
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "base"}]
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{
                "context": context,
                "spill_required": True,
                "spill_failure_action": "Spill failed.",
            }],
        ),
        patch(
            "tools.hook_output_spill.write_spill_file",
            return_value={"ok": False, "path": None, "error": "ENOSPC"},
        ),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("terminal", "call-1")],
            effective_task_id="task-mandatory-failure-budget",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=80, preview_size=20),
        )

    assert "mandatory spill failed" in messages[0]["content"]
    assert context in messages[0]["content"]
    assert "Spill failed." in messages[0]["content"]


def test_untrusted_failure_phrase_does_not_bypass_hook_budget():
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-untrusted-failure-phrase",
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "base"}]
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "mandatory spill failed " + "x" * 1_000}],
        ),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("terminal", "call-1")],
            effective_task_id="task-untrusted-failure-phrase",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=100, preview_size=20),
        )

    assert "x" * 500 not in messages[0]["content"]
    assert "_mandatory_spill_failure" not in messages[0]


def test_spoofed_persistence_markers_do_not_bypass_base_rebudgeting():
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-spoofed-budget-marker",
        _apply_pending_internal_events_to_tool_results=lambda *_args: None,
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    spoofed = "<persisted-output>\nTruncated:" + "x" * 1_000
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": spoofed}]

    _finalize_tool_boundary(
        agent,
        messages,
        messages,
        [_tool_call("terminal", "call-1")],
        effective_task_id="task-spoofed-budget-marker",
        api_call_count=1,
        budget=BudgetConfig(turn_budget=100, preview_size=20),
    )

    assert messages[0]["content"] != spoofed
    assert "x" * 500 not in messages[0]["content"]
    assert "_trusted_budget_evidence" not in messages[0]


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
    assert replay[0]["content"] == "screenshot\n[screenshot]\n\n\nIMAGE-DELTA"
    assert "base64" not in replay[0]["content"]
    db.close()


def test_multimodal_hook_suffix_is_aggregate_budgeted(tmp_path):
    agent = SimpleNamespace(
        session_id=None,
        _session_db=None,
        _current_turn_id="turn-multimodal-budget",
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    original = [
        {"type": "text", "text": "screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    messages = [
        {"role": "tool", "tool_call_id": "call-image", "content": original}
    ]
    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.lifecycle.invoke_hook",
            return_value=[{"context": "MULTIMODAL-HOOK-" + "x" * 20_000}],
        ),
    ):
        _finalize_tool_boundary(
            agent,
            messages,
            messages,
            [_tool_call("computer_use", "call-image")],
            effective_task_id="task-multimodal-budget",
            api_call_count=1,
            budget=BudgetConfig(turn_budget=200, preview_size=40),
        )

    assert messages[0]["content"][:2] == original
    hook_text = messages[0]["content"][-1]["text"]
    assert "x" * 1_000 not in hook_text
    assert len(hook_text) < 200
    assert len("".join(
        part.get("text", "")
        for part in messages[0]["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    )) <= 200


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


def test_session_db_tool_result_batch_rolls_back_on_missing_call_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "tool", content="old-1", tool_call_id="call-1")
    db.append_message("s1", "tool", content="old-2", tool_call_id="call-2")

    assert db.set_tool_result_contents(
        "s1", [("call-1", "new-1"), ("missing", "new-missing")]
    ) == 0
    rows = db.get_messages_as_conversation("s1")
    assert [row["content"] for row in rows] == ["old-1", "old-2"]
    db.close()


def test_session_db_tool_result_batch_rejects_stale_turn_lease(tmp_path):
    from hermes_state import SessionTurnLeaseLostError

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "tool", content="new-owner", tool_call_id="call-0")
    assert db.try_acquire_session_turn_lease("s1", "old-holder") is True
    db.release_session_turn_lease("s1", "old-holder")
    assert db.try_acquire_session_turn_lease("s1", "new-holder") is True

    with pytest.raises(SessionTurnLeaseLostError):
        db.set_tool_result_contents(
            "s1",
            [("call-0", "stale-owner")],
            turn_lease_holder="old-holder",
        )

    assert db.get_messages_as_conversation("s1")[0]["content"] == "new-owner"
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
        def set_tool_result_contents(self, *_args):
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
