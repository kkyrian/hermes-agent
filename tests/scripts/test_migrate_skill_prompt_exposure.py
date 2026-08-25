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


@pytest.mark.parametrize(
    "content",
    (
        "skills: {}\n",
        "skills:\n  prompt_exposure: []\n",
    ),
)
def test_policy_fragment_requires_explicit_mapping(tmp_path, content):
    policy = tmp_path / "policy.yaml"
    policy.write_text(content, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="skills.prompt_exposure mapping",
    ):
        MOD._policy_from_fragment(policy)


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


def test_rollback_preserves_unrelated_config_edits(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        "model:\n  default: before\nskills:\n  other: retained\n",
        encoding="utf-8",
    )
    policy = tmp_path / "policy.yaml"
    _policy(policy)

    applied = MOD.apply_policy(home, policy, apply=True, allow_live=False)
    current = yaml.safe_load(config.read_text(encoding="utf-8"))
    current["model"]["default"] = "after"
    current["skills"]["later"] = True
    config.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")

    MOD.rollback(Path(applied["manifest"]), apply=True, allow_live=False)

    rolled_back = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert rolled_back["model"]["default"] == "after"
    assert rolled_back["skills"] == {"other": "retained", "later": True}


def test_rollback_refuses_changed_prompt_exposure(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    applied = MOD.apply_policy(home, policy, apply=True, allow_live=False)
    config = home / "config.yaml"
    current = yaml.safe_load(config.read_text(encoding="utf-8"))
    current["skills"]["prompt_exposure"]["default"] = "description"
    config.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="changed after apply"):
        MOD.rollback(Path(applied["manifest"]), apply=True, allow_live=False)


def test_rapid_successive_applies_use_distinct_backup_directories(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("model:\n  default: test\n", encoding="utf-8")
    first_policy = tmp_path / "first.yaml"
    second_policy = tmp_path / "second.yaml"
    _policy(first_policy)
    _policy(second_policy)
    second_policy.write_text(
        second_policy.read_text(encoding="utf-8").replace(
            "default: name", "default: description"
        ),
        encoding="utf-8",
    )

    fixed_now = mock.Mock()
    fixed_now.strftime.return_value = "20260823T120000000000Z"
    with mock.patch.object(MOD.dt, "datetime") as datetime_mock:
        datetime_mock.now.return_value = fixed_now
        first = MOD.apply_policy(home, first_policy, apply=True, allow_live=False)
        second = MOD.apply_policy(home, second_policy, apply=True, allow_live=False)

    assert Path(first["manifest"]).parent != Path(second["manifest"]).parent
    assert Path(first["manifest"]).exists()
    assert Path(second["manifest"]).exists()


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


def test_manifest_write_tolerates_unsupported_directory_fsync(monkeypatch, tmp_path):
    target = tmp_path / "manifest.json"
    real_open = MOD.os.open

    def _open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise OSError("directory descriptors unsupported")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(MOD.os, "open", _open)

    MOD._atomic_json_write(target, {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


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


def test_symlinked_named_profile_requires_two_explicit_flags(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    live_root = fake_home / ".hermes" / "profiles"
    live_root.mkdir(parents=True)
    external = tmp_path / "external-profile"
    external.mkdir()
    profile = live_root / "linked"
    profile.symlink_to(external, target_is_directory=True)
    policy = tmp_path / "policy.yaml"
    _policy(policy)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    with pytest.raises(PermissionError):
        MOD.apply_policy(profile, policy, apply=True, allow_live=False)

    assert not (external / "config.yaml").exists()


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


@pytest.mark.parametrize("field", ("config", "backup_config"))
def test_rollback_rejects_manifest_paths_outside_owned_locations(tmp_path, field):
    home = tmp_path / "profile"
    home.mkdir()
    manifest_dir = home / "backups" / "skill-prompt-exposure" / "stamp"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "home": str(home),
        "config": str(home / "config.yaml"),
        "backup_config": str(manifest_dir / "config.yaml.before"),
        "config_existed": True,
    }
    manifest[field] = str(tmp_path / "outside.yaml")
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        MOD.rollback(manifest_path, apply=False, allow_live=False)


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
