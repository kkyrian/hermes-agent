---
title: "fix: Preserve compression routing and identify delegated results"
status: active
date: 2026-08-09
type: fix
target_repo: hermes-agent
origin: Agent Observatory first-live-Hermes review, settled items 1, 2, 7, and 15
---

# fix: Preserve compression routing and identify delegated results

## Summary

Correct three Hermes-owned contracts without changing their established control flow:

1. YAML `null` in `auxiliary.<task>` must remain absent rather than becoming the literal string `"None"`, so compression continues to inherit the current parent provider/model.
2. Compression attempt telemetry must preserve the completed attempt's fallback verdict across session-boundary callbacks.
3. Provider-visible single and batch background-delegation completions must identify themselves, in their own text, as Hermes-generated internal evidence rather than user input.

The work preserves native deterministic compression fallback, current-parent routing, durable delegation storage, owner-session routing, restart restoration, and caller-selected single versus batch delegation.

## Implementation units

### U1. Normalize optional auxiliary configuration

- Normalize `None`, empty strings, and whitespace-only optional task configuration fields to absence before provider/model resolution.
- Preserve the existing `auto` sentinel behavior and explicit caller overrides.
- Add a regression proving `auxiliary.compression.model: null` does not override current-parent routing.

### U2. Preserve truthful attempt telemetry

- Snapshot content-free attempt telemetry and the fallback verdict before rotation/session callbacks can clear compressor state.
- Emit the snapshot only after the existing boundary result is known.
- Add a regression where a boundary callback clears compressor state but the emitted attempt still reports `fallback_used: true` and retains auxiliary-attempt metadata.

### U3. Patch the exact delegation completion envelope

- Change the shared formatter used by TUI and gateway delivery.
- Begin both single and batch text with a literal `HERMES INTERNAL EVENT` / `SUBAGENT RESULT` / `NOT USER INPUT` marker.
- Explain in the same provider-visible input that Hermes generated the event after delegated work finished and that the result is evidence for the parent's ongoing task.
- Do not alter event role, persistence, routing, queueing, restoration, or `NO_REPLY` handling.
- Add focused formatter tests for both shapes and keep routing tests green.

## Scope boundaries

- No Handler, deployed-profile, gateway configuration, or live-runtime mutation.
- No change to compression fallback policy or `abort_on_summary_failure`.
- No conversion of completion events to a new provider role or control-flow architecture.
- Item 12 is dropped and receives no implementation.
- Item 15 currently maps to Handler maintenance-report wording, not a concrete Hermes-owned duration surface. No Hermes duration change will be made unless implementation inspection finds a provider-visible Hermes report that makes the same scope error.

## Verification

- `scripts/run_tests.sh tests/agent/test_auxiliary_client.py -k null`
- `scripts/run_tests.sh tests/agent/test_compression_attempt_telemetry.py`
- `scripts/run_tests.sh tests/tools/test_async_delegation.py -k 'reinjection or formatter or batch'`
- Run the complete affected test files after focused failures are resolved.
- Hostile self-review: parent-runtime inheritance, explicit override precedence, callback state clearing, fallback false-positive risk, single/batch text parity, and unchanged durable routing.
