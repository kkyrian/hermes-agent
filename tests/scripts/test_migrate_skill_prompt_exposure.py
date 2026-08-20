import importlib.util
import json
from pathlib import Path
from unittest import mock

import yaml
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "migrate_skill_prompt_exposure.py"
SPEC = importlib.util.spec_from_file_location("migrate_skill_prompt_exposure", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def _policy(path: Path):
    path.write_text(
        "skills:\n  prompt_exposure:\n    default: name\n"
        "    hidden: [drop-me]\n    names_only: [trim-me]\n"
        "    descriptions: [keep-me]\n    conditional: {}\n",
        encoding="utf-8",
    )


def test_dry_run_has_no_side_effects(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("model:\n  default: test\n", encoding="utf-8")
    before = config.read_bytes()
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    report = MOD.apply_policy(home, policy, apply=False, allow_live=False)
    assert report["action"] == "dry-run"
    assert report["counts"] == {
        "hidden": 1, "names_only": 1, "descriptions": 1, "conditional": 0
    }
    assert config.read_bytes() == before
    assert not (home / "backups").exists()


def test_apply_backup_and_rollback(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    original = "model:\n  default: test\n"
    config.write_text(original, encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    _policy(policy)

    applied = MOD.apply_policy(home, policy, apply=True, allow_live=False)
    manifest = Path(applied["manifest"])
    assert manifest.exists()
    assert yaml.safe_load(config.read_text())["skills"]["prompt_exposure"]["default"] == "name"

    dry = MOD.rollback(manifest, apply=False, allow_live=False)
    assert dry["action"] == "rollback-dry-run"
    assert config.read_text() != original

    MOD.rollback(manifest, apply=True, allow_live=False)
    assert config.read_text() == original
    assert json.loads(manifest.read_text())["backup_config"]


def test_manifest_publication_failure_does_not_mutate_config(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    original = "model:\n  default: test\n"
    config.write_text(original, encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    _policy(policy)

    with mock.patch.object(MOD, "_atomic_json_write", side_effect=OSError("disk full")):
        try:
            MOD.apply_policy(home, policy, apply=True, allow_live=False)
        except OSError as error:
            assert str(error) == "disk full"
        else:
            raise AssertionError("manifest failure was not propagated")

    assert config.read_text(encoding="utf-8") == original


def test_live_profile_requires_two_explicit_flags(monkeypatch, tmp_path):
    live_root = tmp_path / ".hermes" / "profiles"
    home = live_root / "example"
    home.mkdir(parents=True)
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    dry_run = MOD.apply_policy(home, policy, apply=False, allow_live=False)
    assert dry_run["action"] == "dry-run"
    assert not (home / "config.yaml").exists()

    for apply, allow in ((True, False),):
        try:
            MOD.apply_policy(home, policy, apply=apply, allow_live=allow)
        except PermissionError:
            pass
        else:
            raise AssertionError("live profile mutation was not refused")

    applied = MOD.apply_policy(home, policy, apply=True, allow_live=True)
    assert Path(applied["manifest"]).exists()


def test_default_home_is_live_for_apply_and_rollback(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    config = home / "config.yaml"
    original = "model:\n  default: test\n"
    config.write_text(original, encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(PermissionError):
        MOD.apply_policy(home, policy, apply=True, allow_live=False)

    applied = MOD.apply_policy(home, policy, apply=True, allow_live=True)
    manifest = Path(applied["manifest"])

    with pytest.raises(PermissionError):
        MOD.rollback(manifest, apply=True, allow_live=False)

    MOD.rollback(manifest, apply=True, allow_live=True)
    assert config.read_text(encoding="utf-8") == original


def test_checked_in_item14_policy_invariants():
    policy_path = Path(__file__).parents[2] / "docs/design/item14-skill-prompt-exposure.yaml"
    policy = yaml.safe_load(policy_path.read_text())["skills"]["prompt_exposure"]
    buckets = [policy["hidden"], policy["names_only"], policy["descriptions"]]
    assert policy["default"] in {"hidden", "name", "description"}
    assert all(isinstance(bucket, list) for bucket in buckets)
    assert all(isinstance(name, str) and name.strip() for bucket in buckets for name in bucket)
    assert all(len(bucket) == len(set(bucket)) for bucket in buckets)
    assert all(isinstance(name, str) and name.strip() for name in policy["conditional"])
    assert all(left.isdisjoint(right) for index, left in enumerate(map(set, buckets)) for right in map(set, buckets[index + 1:]))
