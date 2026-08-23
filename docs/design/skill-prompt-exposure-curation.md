# Skill Prompt Exposure Curation

Status: implementation plan (2026-08-09)

## Goal

Make the cold-start skill index profile-configurable without deleting or duplicating skill packages. A managed profile can hide a skill from the initial prompt, show its name only, or show its name and description. `skills_list` remains the complete name-and-description discovery surface and `skill_view` remains the complete body-loading surface.

Concrete managed-profile policy belongs in the private deployment repository. This public design uses anonymized examples only. Workspace-owned skills remain supplied only by the cwd-scoped context plugin; this change does not scan `.claude/skills`, `.agents/skills`, or create another registry.

## Invariants

- Exposure policy changes prompt rendering only. It never deletes, archives, edits, copies, or disables an installed skill.
- The policy is resolved while the system prompt is constructed and is part of the prompt-cache key. A conversation's system prompt remains byte-stable.
- Name-only skills remain visible in the initial index and retain descriptions in `skills_list`.
- Prompt-hidden skills remain available through explicit `skills_list` and `skill_view` calls.
- `skill_view` continues to return the complete `SKILL.md` body and linked-file inventory; no pagination is added.
- An absent policy preserves upstream behavior. Unknown/new skills use the configured `default` tier, with `description` as the backward-compatible default.
- Existing platform, environment, disabled-skill, and tool/toolset conditions run before exposure tiers.

## Configuration contract

Add `skills.prompt_exposure` to `config.yaml`:

```yaml
skills:
  prompt_exposure:
    default: description       # description | name | hidden
    hidden: [skill-a]
    names_only: [skill-b]
    descriptions: [skill-c]
    conditional:
      opencode:
        tier: description
        requires_toolsets: [terminal]
        requires_executables: [opencode]
```

Explicit lists override `default`; duplicate membership is invalid and fails closed to `hidden` with a warning. A conditional entry is an additional cold-start gate: its skill is hidden unless every declared toolset is in the session's already-resolved toolsets and every executable resolves on `PATH`. Presence in the profile configuration is the explicit authority to advertise that external worker. This makes `opencode` visible only when the profile authorizes it and the matching terminal surface and binary are actually available. Explicit loading remains possible because this is an offer-time rule, not a package permission system.

## Implementation sequence

1. Add a small parser/normalizer in `agent/prompt_builder.py` that reads the managed, read-only profile configuration and returns an immutable exposure policy plus a deterministic cache fingerprint.
2. Apply the policy after normal compatibility/condition filtering and before category rendering. Keep hidden entries out of the initial index; render name-tier entries without descriptions even inside ordinary categories.
3. Include the policy fingerprint and executable availability results in the existing skills prompt cache key. Do not rebuild a prompt after conversation construction.
4. Document the new configuration defaults and update prompt-size reporting only if tests show it misstates index costs.
5. Correct any bundled-skill frontmatter that names nonexistent toolsets.
6. Keep only an anonymized example policy in this repository. Concrete policy matrices and profile-local content migrations remain private deployment inputs and must not be copied into this repository.
7. Add a migration helper that targets an explicit `HERMES_HOME`, defaults to a side-effect-free dry-run, backs up `config.yaml` before applying, writes atomically, emits a manifest, and can restore from that manifest. Learned-skill cleanup is a separate reviewed package migration with its own backup because private profile-local bodies must not be copied into this repository. Live-profile writes and rollbacks require both `--apply` and `--allow-live-profile`; tests use temporary homes only.

## Verification plan

- Unit-test all three tiers, precedence, duplicate policy entries, default behavior, category demotion interaction, external skills, platform filtering, and deterministic cache keys.
- Assert the example policy has valid, disjoint exposure buckets.
- In a temporary `HERMES_HOME`, construct representative anonymized skill trees and compare exact prompt bytes across repeated builds.
- Assert hidden skills are absent from cold-start, name-only skills have no description there, retained descriptions are verbatim, `skills_list` still returns descriptions for every loadable skill, and `skill_view` returns full bodies.
- Test `opencode` with all combinations of policy authority, terminal toolset, and executable availability; only the all-true case may advertise it.
- Test bundled-skill metadata corrections against registered toolsets.
- Exercise migration dry-run, backup, atomic apply in a temporary home, injected partial failure, and rollback. Verify the four live profile directories are byte- and mtime-unchanged.
- Record prompt byte/token deltas for representative profiles before and after the policy.

## Hostile review targets

- Cache-key drift or prompt reconstruction inside a conversation.
- Accidental filtering of `skills_list` or `skill_view`.
- A new bundled or learned skill receiving an unsafe implicit tier.
- Name collisions between profile-local, external, org, and plugin skills.
- Workspace KK skills entering Hermes through anything except the Item 6 context plugin.
- Prompt-hidden being misread as package deletion or curator suppression.
- OpenCode false positives from a binary without terminal authority, or authority without a usable binary.
- Partial migration leaving config or learned packages mixed-version; rollback restoring the wrong profile.
- The `research-paper-writing` metadata fix naming a tool rather than the registered `file` toolset.

## Rollout boundary

Implementation and tests do not change a live profile. A later private, reviewed rollout must use the migration helper's dry-run output, take a fresh backup, apply one representative profile, run prompt/exposure probes, then proceed profile by profile with rollback manifests retained.
