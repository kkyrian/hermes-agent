from __future__ import annotations

import threading
from types import SimpleNamespace

import run_agent as _ra
from agent.conversation_loop import (
    _close_stream_before_internal_followup,
    _inject_pending_internal_events_before_stop,
    _inject_pending_internal_events_pre_api,
    _inject_pending_steer_pre_api,
    _rollback_pre_api_internal_projection,
)
from agent.tool_executor import _persist_final_tool_result_batch


def _bare_agent() -> _ra.AIAgent:
    agent = object.__new__(_ra.AIAgent)
    agent._pending_internal_events = []
    agent._pending_internal_events_lock = threading.Lock()
    agent._internal_event_mailbox_open = False
    return agent


class _CapturingSessionDB:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def append_messages_batch(self, session_id, messages, **kwargs):
        for message in messages:
            row = {
                "role": message.get("role"),
                "content": message.get("content"),
                "display_kind": message.get("display_kind"),
            }
            if message.get("tool_call_id") is not None:
                row["tool_call_id"] = message["tool_call_id"]
            self.rows.append(row)
        start = len(self.rows) - len(messages) + 1
        return list(range(start, len(self.rows) + 1))

    def flush_token_counts(self) -> None:
        return None

    def set_tool_result_contents(self, session_id, updates):
        del session_id
        staged = []
        for tool_call_id, content in updates:
            matches = [
                row for row in self.rows if row.get("tool_call_id") == tool_call_id
            ]
            if len(matches) != 1:
                return 0
            staged.append((matches[0], content))
        for row, content in staged:
            row["content"] = content
        return len(staged)


def test_internal_event_mailbox_accepts_and_drains_once_in_order() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()

    assert agent.enqueue_internal_event("first") is True
    assert agent.enqueue_internal_event("second") is True

    assert agent._take_internal_events_at_boundary() == ["first", "second"]
    assert agent._take_internal_events_at_boundary() == []


def test_final_boundary_closes_empty_mailbox_without_losing_racing_event() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()

    assert agent.enqueue_internal_event("arrived-before-final") is True
    assert agent._take_internal_events_at_boundary(close_if_empty=True) == [
        "arrived-before-final"
    ]
    assert agent.enqueue_internal_event("arrived-after-drain") is True

    assert agent._take_internal_events_at_boundary(close_if_empty=True) == [
        "arrived-after-drain"
    ]
    assert agent._take_internal_events_at_boundary(close_if_empty=True) == []
    assert agent.enqueue_internal_event("too-late") is False


def test_final_boundary_and_gateway_admission_are_atomic_under_race() -> None:
    for _ in range(100):
        agent = _bare_agent()
        agent._open_internal_event_mailbox()
        start = threading.Barrier(3)
        outcome: dict[str, object] = {}

        def enqueue() -> None:
            start.wait()
            outcome["accepted"] = agent.enqueue_internal_event("racing completion")

        def close_boundary() -> None:
            start.wait()
            outcome["drained"] = agent._take_internal_events_at_boundary(
                close_if_empty=True
            )

        enqueue_thread = threading.Thread(target=enqueue)
        close_thread = threading.Thread(target=close_boundary)
        enqueue_thread.start()
        close_thread.start()
        start.wait()
        enqueue_thread.join(timeout=1)
        close_thread.join(timeout=1)

        assert enqueue_thread.is_alive() is False
        assert close_thread.is_alive() is False
        accepted = outcome["accepted"]
        drained = outcome["drained"]
        assert accepted is False or drained == ["racing completion"]


def test_internal_event_mailbox_rejects_when_turn_is_not_accepting() -> None:
    agent = _bare_agent()

    assert agent.enqueue_internal_event("idle-delivery-fallback") is False


def test_codex_app_server_rejects_mailbox_it_does_not_consume() -> None:
    agent = _bare_agent()
    setattr(agent, "api_mode", "codex_app_server")
    agent._open_internal_event_mailbox()

    assert agent.enqueue_internal_event("use typed idle fallback") is False


def test_closing_internal_event_mailbox_returns_undelivered_events() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("preserve on abnormal exit") is True

    assert agent._close_internal_event_mailbox() == ["preserve on abnormal exit"]
    assert agent.enqueue_internal_event("idle fallback") is False


def test_early_return_attaches_undelivered_internal_events() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("completion before direct return")

    result = agent._finalize_internal_event_mailbox(
        {"final_response": "provider error", "messages": []}
    )

    assert result["pending_internal_events"] == [
        "completion before direct return"
    ]
    assert agent.enqueue_internal_event("too late") is False


def test_exception_exit_preserves_events_for_gateway_harvest() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("completion before exception")

    assert agent._finalize_internal_event_mailbox(None) is None
    assert agent._take_undelivered_internal_events_after_turn() == [
        "completion before exception"
    ]
    assert agent._take_undelivered_internal_events_after_turn() == []


def test_tool_boundary_delivers_pending_internal_events_once() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("event one") is True
    assert agent.enqueue_internal_event("event two") is True
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
    ]

    agent._apply_pending_internal_events_to_tool_results(messages, 1)

    assert len(messages) == 2
    assert messages[-1]["role"] == "tool"
    content = messages[-1]["content"]
    assert content.startswith("tool output")
    assert content.index("event one") < content.index("event two")
    assert content.count("event one") == 1
    assert content.count("event two") == 1
    assert agent._take_internal_events_at_boundary() == []


def test_tool_boundary_internal_event_survives_durable_replay() -> None:
    agent = _bare_agent()
    setattr(agent, "_persist_user_message_idx", None)
    setattr(agent, "_persist_user_message_override", None)
    db = _CapturingSessionDB()
    setattr(agent, "_session_db", db)
    setattr(agent, "_session_db_created", True)
    setattr(agent, "_last_flushed_db_idx", 0)
    setattr(agent, "session_id", "sess-internal-event")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("durable completion evidence")

    assert agent._apply_pending_internal_events_to_tool_results(messages, 1)
    _persist_final_tool_result_batch(agent, messages[-1:])

    rows = db.rows
    assert rows[-1]["role"] == "tool"
    assert rows[-1]["content"] == "tool output\n\ndurable completion evidence"


def test_hidden_internal_context_is_provider_valid_after_tool_result() -> None:
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        {
            "role": "user",
            "content": "completion evidence",
            "_internal_event_synthetic": True,
            "display_kind": "hidden",
        },
    ]
    original = [message.copy() for message in messages]

    assert agent._repair_message_sequence(messages) == 0
    assert messages == original


def test_user_steer_remains_later_and_higher_authority_at_tool_boundary() -> None:
    agent = _bare_agent()
    setattr(agent, "_pending_steer", "user correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("subagent evidence")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
    ]

    agent._apply_pending_internal_events_to_tool_results(messages, 1)
    agent._apply_pending_steer_to_tool_results(messages, 1)

    content = messages[-1]["content"]
    assert content.index("subagent evidence") < content.index("user correction")
    assert messages[-1]["role"] == "tool"


def test_late_steer_after_internal_boundary_remains_later_and_durable() -> None:
    agent = _bare_agent()
    setattr(agent, "_pending_steer", "late owner correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    setattr(agent, "_persist_user_message_idx", None)
    setattr(agent, "_persist_user_message_override", None)
    db = _CapturingSessionDB()
    setattr(agent, "_session_db", db)
    setattr(agent, "_session_db_created", True)
    setattr(agent, "_last_flushed_db_idx", 0)
    setattr(agent, "session_id", "sess-late-steer")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1"}, {"id": "call-2"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        {"role": "tool", "tool_call_id": "call-2", "content": "tool output"},
        {
            "role": "user",
            "content": "internal evidence",
            "_internal_event_synthetic": True,
            "display_kind": "hidden",
        },
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])

    agent._apply_pending_steer_to_tool_results(messages, 2)
    assert agent._pending_steer == "late owner correction"

    assert _inject_pending_steer_pre_api(
        agent, messages, conversation_history=[]
    )

    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["display_kind"] == "hidden"
    assert "late owner correction" in messages[-1]["content"]
    assert agent._repair_message_sequence(messages) == 0
    assert db.rows[-1]["content"] == messages[-1]["content"]


def test_pre_api_boundary_merges_late_completion_into_internal_context() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("second completion")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "first completion"},
    ]

    assert _inject_pending_internal_events_pre_api(agent, messages)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "first completion\n\nsecond completion"
    assert agent._repair_message_sequence(messages) == 0
    assert agent._take_internal_events_at_boundary() == []
    assert agent._internal_event_projection_awaiting_sample is True


def test_pre_api_projection_rollback_restores_sample_marker() -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("late completion")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
    ]

    assert _inject_pending_internal_events_pre_api(agent, messages)
    assert agent._internal_event_projection_awaiting_sample is True

    _rollback_pre_api_internal_projection(agent)

    assert agent._internal_event_projection_awaiting_sample is False
    assert agent._take_internal_events_at_boundary() == ["late completion"]


def test_pre_api_boundary_updates_an_already_persisted_tool_carrier() -> None:
    agent = _bare_agent()
    setattr(agent, "_persist_user_message_idx", None)
    setattr(agent, "_persist_user_message_override", None)
    db = _CapturingSessionDB()
    setattr(agent, "_session_db", db)
    setattr(agent, "_session_db_created", True)
    setattr(agent, "_last_flushed_db_idx", 0)
    setattr(agent, "session_id", "sess-late-internal-event")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "first completion"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("late completion")

    assert _inject_pending_internal_events_pre_api(agent, messages)

    assert messages[-1]["_db_persisted"] is True
    assert db.rows[-1]["content"] == "first completion\n\nlate completion"


def test_late_steer_requeues_when_persisted_carrier_update_fails() -> None:
    agent = _bare_agent()
    setattr(agent, "_pending_steer", "owner correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    setattr(agent, "_session_db", SimpleNamespace(
        set_tool_result_contents=lambda *_args: 0
    ))
    setattr(agent, "session_id", "sess-failed-steer")
    messages = [{
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "durable tool output",
        "_db_persisted": True,
    }]

    assert not _inject_pending_steer_pre_api(agent, messages)
    assert messages[0]["content"] == "durable tool output"
    assert agent._drain_pending_steer() == "owner correction"


def test_pre_api_steer_does_not_rewrite_tool_from_earlier_turn() -> None:
    agent = _bare_agent()
    setattr(agent, "_pending_steer", "current correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "old result", "_db_persisted": True},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
        {"role": "assistant", "content": "retrying"},
    ]

    assert not _inject_pending_steer_pre_api(agent, messages)
    assert messages[2]["content"] == "old result"
    assert agent._drain_pending_steer() == "current correction"


def test_pre_api_persistence_failure_requeues_event_and_steer(monkeypatch) -> None:
    agent = _bare_agent()
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("late completion")
    setattr(agent, "_pending_steer", "owner correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "first completion",
            "_db_persisted": True,
        },
    ]

    def _fail_persistence(_agent, _messages):
        _agent._incremental_persistence_failed = True

    monkeypatch.setattr(
        "agent.tool_executor._persist_final_tool_result_batch",
        _fail_persistence,
    )

    assert _inject_pending_internal_events_pre_api(agent, messages)
    assert agent._incremental_persistence_failed is True
    _rollback_pre_api_internal_projection(agent)

    assert messages[-1]["content"] == "first completion"
    assert agent._take_internal_events_at_boundary() == ["late completion"]
    assert agent._drain_pending_steer() == "owner correction"


def test_pre_api_boundary_requeues_when_no_safe_carrier_exists() -> None:
    agent = _bare_agent()
    setattr(agent, "_pending_steer", "owner correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("completion evidence")
    messages = [{"role": "user", "content": "original request"}]

    assert not _inject_pending_internal_events_pre_api(agent, messages)
    assert agent._take_internal_events_at_boundary() == ["completion evidence"]
    assert agent._drain_pending_steer() == "owner correction"


def test_final_boundary_continues_once_for_pending_internal_events() -> None:
    agent = _bare_agent()
    setattr(agent, "_budget_grace_call", False)
    setattr(agent, "_session_messages", [])
    setattr(agent, "_api_call_count", 60)
    setattr(agent, "max_iterations", 60)
    setattr(agent, "iteration_budget", SimpleNamespace(remaining=0))
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event(
        "[HERMES INTERNAL EVENT — SUBAGENT RESULT — NOT USER INPUT] result"
    )

    messages: list[dict] = [{"role": "user", "content": "work"}]
    final_msg = {"role": "assistant", "content": "premature final"}

    assert not _inject_pending_internal_events_before_stop(agent, messages, final_msg)
    assert messages == [{"role": "user", "content": "work"}]
    assert agent._take_internal_events_at_boundary() == [
        "[HERMES INTERNAL EVENT — SUBAGENT RESULT — NOT USER INPUT] result"
    ]

    # The first empty final boundary closes admission atomically. A completion
    # arriving afterward must take the gateway's normal idle follow-up path.
    assert not _inject_pending_internal_events_before_stop(
        agent,
        messages,
        {"role": "assistant", "content": "actual final"},
    )
    assert agent.enqueue_internal_event("too late") is False


def test_final_boundary_defers_even_when_budget_is_available() -> None:
    agent = _bare_agent()
    setattr(agent, "_budget_grace_call", False)
    setattr(agent, "_session_messages", [])
    setattr(agent, "_api_call_count", 2)
    setattr(agent, "max_iterations", 60)
    setattr(agent, "iteration_budget", SimpleNamespace(remaining=58))
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("completion")

    assert not _inject_pending_internal_events_before_stop(
        agent,
        [{"role": "user", "content": "work"}],
        {"role": "assistant", "content": "premature final"},
    )
    assert getattr(agent, "_budget_grace_call") is False
    assert agent._take_internal_events_at_boundary() == ["completion"]


def test_streamed_final_is_closed_before_internal_event_resampling() -> None:
    agent = _bare_agent()
    boundaries = []
    agent.stream_delta_callback = boundaries.append
    agent._interim_content_was_streamed = lambda text: text == "visible final"
    agent._stream_needs_break = False

    _close_stream_before_internal_followup(agent, "visible final")

    assert boundaries == [None]
    assert agent._stream_needs_break is True


def test_final_boundary_internal_context_is_returned_for_typed_followup() -> None:
    agent = _bare_agent()
    setattr(agent, "_budget_grace_call", False)
    setattr(agent, "_session_messages", [])
    setattr(agent, "_api_call_count", 2)
    setattr(agent, "max_iterations", 60)
    setattr(agent, "iteration_budget", SimpleNamespace(remaining=58))
    setattr(agent, "_persist_user_message_idx", None)
    setattr(agent, "_persist_user_message_override", None)
    db = _CapturingSessionDB()
    setattr(agent, "_session_db", db)
    setattr(agent, "_session_db_created", True)
    setattr(agent, "_last_flushed_db_idx", 0)
    setattr(agent, "session_id", "sess-final-internal-event")
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("final-race completion")
    messages = [{"role": "user", "content": "work"}]

    assert not _inject_pending_internal_events_before_stop(
        agent,
        messages,
        {"role": "assistant", "content": "premature final"},
    )
    result = agent._finalize_internal_event_mailbox({"final_response": "done"})

    assert result["pending_internal_events"] == ["final-race completion"]
    assert db.rows == []


def test_final_boundary_leaves_user_steer_for_its_authoritative_path() -> None:
    agent = _bare_agent()
    setattr(agent, "_budget_grace_call", False)
    setattr(agent, "_session_messages", [])
    setattr(agent, "_api_call_count", 2)
    setattr(agent, "max_iterations", 60)
    setattr(agent, "iteration_budget", SimpleNamespace(remaining=58))
    setattr(agent, "_pending_steer", "owner correction")
    setattr(agent, "_pending_steer_lock", threading.Lock())
    agent._open_internal_event_mailbox()
    assert agent.enqueue_internal_event("subagent evidence")

    messages = [{"role": "user", "content": "work"}]
    assert not _inject_pending_internal_events_before_stop(
        agent,
        messages,
        {"role": "assistant", "content": "premature final"},
    )

    assert agent._take_internal_events_at_boundary() == ["subagent evidence"]
    assert agent._drain_pending_steer() == "owner correction"
