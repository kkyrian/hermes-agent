"""Regression coverage for CLI async-delegation completion ownership."""

import queue
from unittest.mock import MagicMock

from cli import HermesCLI


def test_cli_completion_drain_uses_visible_session_identity(monkeypatch):
    """A CLI window must not claim another window's restored completion."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_visible",
        "session_key": "visible-session",
    }
    calls = []

    class FakeRegistry:
        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

    claimed = []
    completed = []

    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry(),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: claimed.append((evt, consumer)) or "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-idle")

    assert calls == [("visible-session", True)]
    assert cli._pending_input.get_nowait() == "completion payload"
    assert claimed == [(event, "cli-idle")]
    assert completed == [(event, "claim-token")]


def test_cli_completion_drain_admits_to_active_parent_mailbox(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    active_agent = MagicMock()
    active_agent.enqueue_internal_event.return_value = True
    cli.agent = active_agent
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_active",
        "session_key": "visible-session",
    }

    class FakeRegistry:
        def drain_notifications(self, *, session_key="", owns_event=None):
            assert session_key == "visible-session"
            assert callable(owns_event)
            assert owns_event(event)
            return [(event, "active completion payload")]

    completed = []
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry(),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: "active-claim",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-active")

    active_agent.enqueue_internal_event.assert_called_once_with(
        "active completion payload"
    )
    assert cli._pending_input.empty()
    assert completed == []

    cli._settle_admitted_process_notifications(
        {"final_response": "used completion"}
    )
    assert completed == [(event, "active-claim")]


def test_cli_completion_leftover_is_requeued_before_acknowledgement(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    active_agent = MagicMock()
    active_agent.enqueue_internal_event.return_value = True
    cli.agent = active_agent
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_leftover",
        "session_key": "visible-session",
    }

    class FakeRegistry:
        def drain_notifications(self, *, session_key="", owns_event=None):
            return [(event, "leftover completion")]

    completed = []
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: "leftover-claim",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-active")
    result = {"pending_internal_events": ["leftover completion"]}
    cli._settle_admitted_process_notifications(result)

    assert cli._pending_input.get_nowait() == "leftover completion"
    assert "pending_internal_events" not in result
    assert completed == [(event, "leftover-claim")]


def test_cli_completion_claim_is_released_when_turn_has_no_result(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    event = {"delegation_id": "deleg_error"}
    cli._admitted_process_notification_claims = [
        (event, "error-claim", "completion")
    ]
    released = []
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda evt, token: released.append((evt, token)),
    )

    cli._settle_admitted_process_notifications(None)

    assert released == [(event, "error-claim")]
    assert cli._admitted_process_notification_claims == []


def test_cli_completion_ownership_rejects_foreign_session():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None

    assert not cli._owns_process_notification(
        {"type": "async_delegation", "session_key": "foreign-session"}
    )


def test_cli_completion_ownership_accepts_compression_lineage():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            assert session_id == "pre-compression-session"
            return "visible-session"

    cli._session_db = FakeSessionDB()

    assert cli._owns_process_notification(
        {
            "type": "async_delegation",
            "session_key": "pre-compression-session",
        }
    )
