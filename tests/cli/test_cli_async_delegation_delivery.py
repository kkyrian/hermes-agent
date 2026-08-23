"""Regression coverage for CLI async-delegation completion ownership."""

import queue
from unittest.mock import MagicMock

from cli import HermesCLI, _deferred_completion_turn_is_durable


def test_deferred_completion_ack_requires_durable_turn():
    cli = HermesCLI.__new__(HermesCLI)
    cli._last_chat_turn_durable = False
    assert not _deferred_completion_turn_is_durable(cli, "Error: persistence failed")
    cli._last_chat_turn_durable = True
    assert _deferred_completion_turn_is_durable(cli, "completed")


def test_completion_wrapper_reports_durable_ack_result(monkeypatch):
    from tools.async_delegation import complete_event_delivery

    monkeypatch.setattr(
        "tools.async_delegation.complete_completion_delivery",
        lambda delegation_id, claim: (delegation_id, claim) == ("deleg-1", "claim-1"),
    )

    assert complete_event_delivery(
        {"type": "async_delegation", "delegation_id": "deleg-1"},
        "claim-1",
    ) is True


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
    queued = cli._pending_input.get_nowait()
    assert str(queued) == "completion payload"
    assert queued.event is event
    assert queued.claim == "claim-token"
    assert claimed == [(event, "cli-idle")]
    assert completed == []


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
    released = []
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: "leftover-claim",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda evt, token: released.append((evt, token)),
    )

    cli._drain_process_notifications("cli-active")
    result = {"pending_internal_events": ["leftover completion"]}
    cli._settle_admitted_process_notifications(result)

    queued = cli._pending_input.get_nowait()
    assert str(queued) == "leftover completion"
    assert queued.claim == "leftover-claim"
    assert "pending_internal_events" not in result
    assert completed == []
    assert released == []


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


def test_cli_failed_turn_harvests_completion_before_acknowledgement(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    event = {"delegation_id": "deleg_failed"}
    cli._admitted_process_notification_claims = [
        (event, "failed-claim", "completion payload")
    ]
    cli._stop_admitted_process_notification_renewal = MagicMock()
    cli.agent = MagicMock()
    cli.agent._take_undelivered_internal_events_after_turn.return_value = [
        "completion payload"
    ]
    completed = []
    released = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda evt, token: released.append((evt, token)),
    )

    cli._settle_admitted_process_notifications({"error": "provider failed"})

    queued = cli._pending_input.get_nowait()
    assert str(queued) == "completion payload"
    assert queued.claim == "failed-claim"
    assert completed == []
    assert released == []


def test_cli_failed_turn_releases_unharvested_completion_claim(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    event = {"delegation_id": "deleg_persist_failed"}
    cli._admitted_process_notification_claims = [
        (event, "persist-failed-claim", "completion payload")
    ]
    cli._stop_admitted_process_notification_renewal = MagicMock()
    cli.agent = MagicMock()
    cli.agent._take_undelivered_internal_events_after_turn.return_value = []
    released = []
    completed = []
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda evt, token: released.append((evt, token)),
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._settle_admitted_process_notifications(
        {"failed": True, "final_response": ""}
    )

    assert released == [(event, "persist-failed-claim")]
    assert completed == []
    assert cli._pending_input.empty()


def test_cli_active_completion_claims_are_renewed(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    event = {"delegation_id": "deleg_long_turn"}
    cli._admitted_process_notification_claims = [
        (event, "long-turn-claim", "payload")
    ]
    renewed = []
    monkeypatch.setattr(
        "tools.async_delegation.renew_completion_delivery",
        lambda delegation_id, claim: renewed.append((delegation_id, claim)) or True,
    )

    cli._renew_admitted_process_notification_claims_once()

    assert renewed == [("deleg_long_turn", "long-turn-claim")]


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
