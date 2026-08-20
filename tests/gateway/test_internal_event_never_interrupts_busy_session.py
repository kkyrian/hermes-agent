"""Regression tests for internal synthetic events in a busy session.

Reported by @Heeervas (June 2026): a completed asynchronous ``delegate_task``
re-enters the originating gateway session as an internal ``MessageEvent``.
When that session was busy running a
turn, the completion was treated exactly like a user TEXT message and hit the
default ``busy_input_mode='interrupt'`` path — calling
``running_agent.interrupt()`` and aborting the active turn, plus sending a
"⚡ Interrupting current task" ack. The same shape affects background-process
completions (terminal ``notify_on_complete``), which also re-enter as internal
events.

Generic internal events remain silently queued.  Subagent completions are a
separate typed case: the active parent accepts them into its internal-event
mailbox so they are visible at the next safe model boundary without interrupting
the current request or replaying as a second next-turn message.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import (  # noqa: E402
    GatewayRunner,
    _dequeue_typed_internal_followup,
    _merge_agent_undelivered_internal_events,
    _pending_internal_event_from_result,
    _queue_typed_internal_followup,
    _queued_followup_was_interrupted,
)
from run_agent import AIAgent  # noqa: E402


def _make_internal_event(text: str = "[async delegation completed]") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
        internal=True,
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []  # no active subagents at completion time
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    parent.enqueue_internal_event.return_value = True
    return parent


@pytest.mark.asyncio
async def test_internal_event_does_not_interrupt_busy_session() -> None:
    """The async-delegation completion must not abort the active turn."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"  # the default that caused the bug
    adapter = _make_adapter()
    event = _make_internal_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    assert adapter._pending_messages[sk] is event
    # The active turn must survive.
    parent.interrupt.assert_not_called()
    # No "⚡ Interrupting current task" (or any) ack for a synthetic event.
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_generic_internal_event_stays_separate_from_queued_user_text() -> None:
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    internal = _make_internal_event("terminal completion")
    user = _make_internal_event("owner follow-up")
    user.internal = False
    sk = build_session_key(internal.source)
    adapter._pending_messages[sk] = user
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[internal.source.platform] = adapter

    assert await runner._handle_active_session_busy_message(internal, sk) is True
    assert adapter._pending_messages[sk] is user
    assert adapter._pending_messages[sk].text == "owner follow-up"
    assert runner._queued_events[sk] == [internal]


@pytest.mark.asyncio
async def test_subagent_completion_enters_busy_parent_mailbox() -> None:
    """A typed subagent completion is admitted once without a user-facing ack."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    event = _make_internal_event("A delegated child finished.")
    event.metadata["internal_event_kind"] = "subagent_completion"
    event.metadata["delegation_id"] = "deleg_123"
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    parent._active_children = ["still-running-sibling"]
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    parent.enqueue_internal_event.assert_called_once_with(event.text)
    assert event.metadata["durable_delivery_deferred"] is True
    assert event.metadata["durable_delivery_session_key"] == sk
    assert event.metadata["durable_delivery_generation"] >= 0
    parent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_subagent_admission_uses_separate_typed_followup_queue() -> None:
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    event = _make_internal_event("completion after final boundary")
    event.metadata["internal_event_kind"] = "subagent_completion"
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    parent.enqueue_internal_event.return_value = False
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    assert await runner._handle_active_session_busy_message(event, sk) is True
    assert adapter._pending_messages == {}
    assert _dequeue_typed_internal_followup(runner, sk) is event
    adapter._send_with_retry.assert_not_called()


def test_typed_internal_followup_never_merges_with_pending_user_event() -> None:
    runner = _make_runner()
    internal = _make_internal_event("completion evidence")
    user = _make_internal_event("owner follow-up")
    user.internal = False
    sk = build_session_key(internal.source)
    adapter = _make_adapter()
    adapter._pending_messages[sk] = user

    _queue_typed_internal_followup(runner, sk, internal)

    assert adapter._pending_messages[sk] is user
    assert adapter._pending_messages[sk].text == "owner follow-up"
    assert _dequeue_typed_internal_followup(runner, sk) is internal


@pytest.mark.asyncio
async def test_pr007_ordering_reaches_long_parent_before_next_model_decision() -> None:
    """Replay the incident ordering through gateway admission and model boundary."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    event = _make_internal_event(
        "[HERMES INTERNAL EVENT — SUBAGENT RESULT — NOT USER INPUT — deleg_pr007]\n"
        "Review found a blocker before the parent closeout."
    )
    event.metadata["internal_event_kind"] = "subagent_completion"
    sk = build_session_key(event.source)
    parent = object.__new__(AIAgent)
    setattr(parent, "_pending_internal_events", [])
    setattr(parent, "_pending_internal_events_lock", threading.Lock())
    setattr(parent, "_internal_event_mailbox_open", False)
    parent._open_internal_event_mailbox()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    assert await runner._handle_active_session_busy_message(event, sk) is True
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "long-step"}]},
        {
            "role": "tool",
            "tool_call_id": "long-step",
            "content": "long parent step complete",
        },
    ]
    parent._apply_pending_internal_events_to_tool_results(messages, 1)

    assert messages[-1]["content"].count("Review found a blocker") == 1
    assert parent._take_internal_events_at_boundary() == []
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_async_delegation_watcher_sets_structured_internal_event_kind() -> None:
    runner = _make_runner()
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    platform = MagicMock(value="telegram")
    runner.adapters[platform] = adapter
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_typed",
        "platform": "telegram",
        "chat_id": "123",
        "chat_type": "private",
        "user_id": "user1",
    }

    assert await runner._inject_watch_notification("completion evidence", event) is True
    injected = adapter.handle_message.await_args.args[0]
    assert injected.internal is True
    assert injected.metadata["internal_event_kind"] == "subagent_completion"
    assert injected.metadata["delegation_id"] == "deleg_typed"


def test_abnormal_turn_leftover_keeps_typed_internal_idle_delivery() -> None:
    source = _make_internal_event().source

    event = _pending_internal_event_from_result(
        {
            "pending_internal_events": [
                "[HERMES INTERNAL EVENT — SUBAGENT RESULT — NOT USER INPUT] one",
                "[HERMES INTERNAL EVENT — SUBAGENT RESULT — NOT USER INPUT] two",
            ]
        },
        source,
    )

    assert event is not None
    assert event.internal is True
    assert event.source is source
    assert event.text.count("SUBAGENT RESULT") == 2
    assert event.metadata["internal_event_kind"] == "subagent_completion"
    assert event.metadata["delivery"] == "turn_finalize_fallback"


def test_completed_turn_with_typed_followup_is_not_interrupted() -> None:
    """A completion after the final boundary must not suppress first delivery."""
    result = {"interrupted": False, "final_response": "first response"}

    assert _queued_followup_was_interrupted(result) is False


def test_real_interrupt_still_suppresses_queued_first_delivery() -> None:
    result = {"interrupted": True, "final_response": "interrupted noise"}

    assert _queued_followup_was_interrupted(result) is True


def test_gateway_harvests_exception_path_internal_events() -> None:
    agent = object.__new__(AIAgent)
    setattr(agent, "_pending_internal_events", [])
    setattr(agent, "_pending_internal_events_lock", threading.Lock())
    setattr(agent, "_internal_event_mailbox_open", False)
    setattr(
        agent,
        "_undelivered_internal_events_after_turn",
        ["completion preserved across exception"],
    )

    result = _merge_agent_undelivered_internal_events(None, agent)

    assert result["pending_internal_events"] == [
        "completion preserved across exception"
    ]
    assert agent._take_undelivered_internal_events_after_turn() == []
