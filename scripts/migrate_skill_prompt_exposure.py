#!/usr/bin/env python3
"""Safely apply or roll back a skills.prompt_exposure config fragment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml


def _live_profile_root() -> Path:
    """Return the HOME-anchored profile root used by Hermes profile commands."""
    return Path.home() / ".hermes" / "profiles"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _atomic_yaml_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, tmp_name)
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on hosts that support directory fds."""
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _is_live_profile(home: Path) -> bool:
    default_home = Path.home() / ".hermes"
    try:
        resolved = home.resolve()
        if resolved == default_home.resolve():
            return True
        resolved.relative_to(_live_profile_root().resolve())
        return True
    except ValueError:
        return False


def _policy_from_fragment(path: Path) -> dict:
    fragment = _load_yaml(path)
    policy = ((fragment.get("skills") or {}).get("prompt_exposure") or {})
    if not isinstance(policy, dict):
        raise ValueError("policy fragment must contain skills.prompt_exposure mapping")
    return policy


def _counts(policy: dict) -> dict[str, int]:
    return {
        "hidden": len(policy.get("hidden") or []),
        "names_only": len(policy.get("names_only") or []),
        "descriptions": len(policy.get("descriptions") or []),
        "conditional": len(policy.get("conditional") or {}),
    }


def apply_policy(home: Path, fragment: Path, *, apply: bool, allow_live: bool) -> dict:
    home = home.expanduser().resolve()
    if _is_live_profile(home) and apply and not allow_live:
        raise PermissionError(
            "live profile mutation requires --apply and --allow-live-profile"
        )
    config_path = home / "config.yaml"
    config_existed = config_path.exists()
    config = _load_yaml(config_path)
    policy = _policy_from_fragment(fragment)
    report = {
        "action": "apply" if apply else "dry-run",
        "home": str(home),
        "config": str(config_path),
        "policy": str(fragment.resolve()),
        "counts": _counts(policy),
        "changed": (config.get("skills") or {}).get("prompt_exposure") != policy,
    }
    if not apply or not report["changed"]:
        return report

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = home / "backups" / "skill-prompt-exposure" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_config = backup_dir / "config.yaml.before"
    if config_path.exists():
        shutil.copy2(config_path, backup_config)
    else:
        backup_config.write_text("", encoding="utf-8")

    manifest = {
        **report,
        "backup_config": str(backup_config),
        "config_existed": config_existed,
    }
    manifest_path = backup_dir / "manifest.json"
    _atomic_json_write(manifest_path, manifest)

    updated = dict(config)
    skills = dict(updated.get("skills") or {})
    skills["prompt_exposure"] = policy
    updated["skills"] = skills
    _atomic_yaml_write(config_path, updated)

    report["manifest"] = str(manifest_path)
    return report


def rollback(manifest_path: Path, *, apply: bool, allow_live: bool) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    home = Path(manifest["home"]).resolve()
    if _is_live_profile(home) and apply and not allow_live:
        raise PermissionError(
            "live profile rollback mutation requires --apply and --allow-live-profile"
        )
    config_path = Path(manifest["config"]).resolve()
    expected_config = (home / "config.yaml").resolve()
    if config_path != expected_config:
        raise ValueError("manifest config must be <home>/config.yaml")
    backup_config = Path(manifest["backup_config"]).resolve()
    try:
        backup_config.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ValueError("manifest backup_config must stay within the manifest directory") from exc
    report = {
        "action": "rollback" if apply else "rollback-dry-run",
        "home": str(home),
        "config": str(config_path),
        "manifest": str(manifest_path.resolve()),
    }
    if not apply:
        return report
    if manifest.get("config_existed"):
        _atomic_copy(backup_config, config_path)
    else:
        config_path.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--rollback", type=Path, metavar="MANIFEST")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-live-profile", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        result = rollback(
            args.rollback, apply=args.apply, allow_live=args.allow_live_profile
        )
    else:
        if not args.home or not args.policy:
            parser.error("--home and --policy are required unless --rollback is used")
        result = apply_policy(
            args.home,
            args.policy,
            apply=args.apply,
            allow_live=args.allow_live_profile,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
