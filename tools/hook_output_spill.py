"""Spill oversized hook-injected context to disk with a preview placeholder.

Ported from openai/codex PR #21069 (``Spill large hook outputs from context``).

Background
----------
Both shell hooks (``agent/shell_hooks.py``) and Python plugins
(``pre_llm_call`` hook in ``run_agent.py``) can return ``{"context": "..."}``
which gets concatenated into the current turn's user message on EVERY
subsequent API call. If a hook emits a large blob (e.g. a debug dump, a
full file, or a runaway prompt-engineering script), that blob inflates
every turn of the session and blows out the prompt cache prefix the
moment it's appended.

This mirrors what Codex does for its ``PreToolUse``/``Stop``/feedback
hooks: once the injected text exceeds a configured budget, write the
full content to a per-session directory on disk and replace the in-prompt
payload with a head/tail preview plus the saved path. The model can still
inspect the full content via ``read_file`` or ``terminal`` if it needs to.

Config (``config.yaml``)::

    hooks:
      output_spill:
        enabled: true          # default: true; set false to disable spilling
        max_chars: 10000       # default; context above this is spilled
        preview_head: 500      # chars shown at the start of the preview
        preview_tail: 500      # chars shown at the end of the preview
        directory: null        # default: <HERMES_HOME>/hook_outputs

Design invariants
-----------------
* Behaviour-preserving when ``enabled: false`` or when content is under
  the cap — return the input string unchanged.
* Never raises. Any I/O error (disk full, permission denied, missing
  HERMES_HOME, etc.) falls back to a byte-length truncation with an
  in-prompt notice — the hook context still reaches the model, just
  bounded in size.
* Spill files are grouped by session so a ``/new`` session doesn't grow
  them forever in one directory.
"""

from __future__ import annotations

import logging
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def _coerce_non_negative_int(value: Any, default: int) -> int:
    """Like ``_coerce_positive_int`` but allows zero (e.g. empty tail)."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv < 0:
        return default
    return iv


def get_spill_config() -> Dict[str, Any]:
    """Return resolved hook output-spill config. Never raises."""
    section: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        hooks = cfg.get("hooks") if isinstance(cfg, dict) else None
        if isinstance(hooks, dict):
            sub = hooks.get("output_spill")
            if isinstance(sub, dict):
                section = sub
    except Exception:
        section = {}

    enabled_raw = section.get("enabled", DEFAULT_ENABLED)
    enabled = bool(enabled_raw) if enabled_raw is not None else DEFAULT_ENABLED

    directory = section.get("directory")
    if directory is not None and not isinstance(directory, str):
        directory = None

    return {
        "enabled": enabled,
        "max_chars": _coerce_positive_int(section.get("max_chars"), DEFAULT_MAX_CHARS),
        "preview_head": _coerce_non_negative_int(
            section.get("preview_head"), DEFAULT_PREVIEW_HEAD
        ),
        "preview_tail": _coerce_non_negative_int(
            section.get("preview_tail"), DEFAULT_PREVIEW_TAIL
        ),
        "directory": directory,
    }


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.replace("..", "_").strip(".")
    return cleaned or fallback


def _resolve_spill_dir(
    directory_override: Optional[str],
    session_id: Optional[str],
    *,
    namespace: str = "hook_outputs",
    profile_local: bool = False,
) -> Path:
    """Return the directory where spill files for this session live."""
    if directory_override and not profile_local:
        base = Path(os.path.expanduser(directory_override))
    else:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home())
        for segment in namespace.replace("\\", "/").split("/"):
            if segment:
                base /= _safe_segment(segment, "spill")

    # Group by session so spills are contained per conversation.
    session_segment = _safe_segment(session_id or "no-session", "no-session")
    return base / session_segment


def _spill_root_and_segments(
    directory_override: Optional[str],
    session_id: Optional[str],
    *,
    namespace: str,
    profile_local: bool,
) -> tuple[Path, list[str]]:
    """Return a trusted root plus untrusted path segments to traverse safely."""
    session_segment = _safe_segment(session_id or "no-session", "no-session")
    if directory_override and not profile_local:
        destination = Path(os.path.expanduser(directory_override)).absolute()
        root = Path(destination.anchor)
        segments = list(destination.parts[1:])
    else:
        from hermes_constants import get_hermes_home

        root = Path(get_hermes_home()).resolve(strict=True)
        segments = [
            _safe_segment(segment, "spill")
            for segment in namespace.replace("\\", "/").split("/")
            if segment
        ]
    segments.append(session_segment)
    return root, segments


def _supports_secure_dir_fd() -> bool:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    return os.name != "nt" and all(hasattr(os, name) for name in required)


def _is_link_or_reparse(path: Path) -> bool:
    """Detect symlinks and Windows junction/reparse-point directories."""
    try:
        if path.is_symlink():
            return True
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except OSError:
        return False


def _write_spill_with_dir_fd(text: str, root: Path, segments: list[str]) -> Path:
    directory_fd: Optional[int] = None
    temp_name: Optional[str] = None
    try:
        directory_fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        for segment in segments:
            try:
                os.mkdir(segment, 0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                segment,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        temp_name = f".spill-{uuid.uuid4().hex}.tmp"
        final_name = f"{uuid.uuid4().hex}.txt"
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            final_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temp_name = None
        os.chmod(final_name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
        return root.joinpath(*segments, final_name)
    finally:
        if directory_fd is not None and temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _write_spill_portable(text: str, root: Path, segments: list[str]) -> Path:
    """Portable containment path for native Windows and limited POSIX hosts."""
    root = root.resolve(strict=True)
    current = root
    for segment in segments:
        candidate = current / segment
        if candidate.exists() or candidate.is_symlink():
            if _is_link_or_reparse(candidate) or not candidate.is_dir():
                raise OSError(f"refusing spill through linked path: {candidate}")
        else:
            candidate.mkdir(mode=0o700)
        if _is_link_or_reparse(candidate):
            raise OSError(f"refusing spill through linked path: {candidate}")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        current = candidate

    temp_path = current / f".spill-{uuid.uuid4().hex}.tmp"
    final_path = current / f"{uuid.uuid4().hex}.txt"
    fd: Optional[int] = None
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)
        os.chmod(final_path, 0o600)
        if _is_link_or_reparse(final_path):
            raise OSError(f"refusing linked spill result: {final_path}")
        final_path.resolve(strict=True).relative_to(root)
        return final_path
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def write_spill_file(
    text: str,
    *,
    session_id: Optional[str],
    namespace: str = "hook_outputs",
    directory_override: Optional[str] = None,
    profile_local: bool = False,
) -> Dict[str, Any]:
    """Atomically persist spill content and return a structured result."""
    try:
        root, segments = _spill_root_and_segments(
            directory_override,
            session_id,
            namespace=namespace,
            profile_local=profile_local,
        )
        if _supports_secure_dir_fd():
            spill_path = _write_spill_with_dir_fd(text, root, segments)
        else:
            spill_path = _write_spill_portable(text, root, segments)
        return {"ok": True, "path": str(spill_path), "error": None}
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        logger.warning("hook output spill failed: %s", exc)
        return {"ok": False, "path": None, "error": str(exc)}


def _build_preview(
    text: str,
    head: int,
    tail: int,
    saved_path: Optional[str],
    *,
    source: str,
    success_action: Optional[str] = None,
    failure_action: Optional[str] = None,
) -> str:
    """Assemble the in-prompt preview with head/tail and saved-path footer."""
    total = len(text)
    head_chunk = text[:head] if head > 0 else ""
    tail_chunk = text[-tail:] if tail > 0 and total > head else ""

    parts = [
        f"[{source} output truncated — {total:,} chars; full content "
        + (f"saved to {saved_path}]" if saved_path else "unavailable — spill write failed]"),
    ]
    if head_chunk:
        parts.append("--- head ---")
        parts.append(head_chunk)
    if tail_chunk:
        parts.append("--- tail ---")
        parts.append(tail_chunk)
    action = success_action if saved_path else failure_action
    if action:
        parts.append(action)
    return "\n".join(parts)


def spill_if_oversized(
    text: str,
    *,
    session_id: Optional[str] = None,
    source: str = "hook",
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    namespace: str = "hook_outputs",
    success_action: Optional[str] = None,
    failure_action: Optional[str] = None,
) -> str:
    """Spill ``text`` to disk if it exceeds the configured cap.

    Returns either ``text`` unchanged (when under the cap, disabled, or
    empty) or a preview string with a filesystem path pointing at the
    full content.

    Parameters
    ----------
    text:
        The raw injected-context string from a hook. Non-string inputs
        are coerced with ``str()``.
    session_id:
        Used to group spill files by conversation. Falls back to
        ``"no-session"`` if missing.
    source:
        Human-readable label used in the preview header (``"hook"``,
        ``"plugin hook"``, ``"shell hook"``, etc.). Free-form.
    config:
        Optional override for tests; normally resolved from
        ``config.yaml``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    cfg = config if config is not None else get_spill_config()
    if not force and not cfg.get("enabled", True):
        return text

    max_chars = int(cfg.get("max_chars") or DEFAULT_MAX_CHARS)
    if not force and len(text) <= max_chars:
        return text

    head = int(cfg.get("preview_head") or 0)
    tail = int(cfg.get("preview_tail") or 0)
    directory_override = cfg.get("directory")

    # Try to write the spill file. If that fails we still need to return
    # something bounded — never let a disk failure blow up the turn.
    result = write_spill_file(
        text,
        session_id=session_id,
        namespace=namespace,
        directory_override=directory_override,
        profile_local=force,
    )
    if force and not result["ok"]:
        parts = [
            f"[{source} mandatory spill failed — {len(text):,} chars; complete "
            "content retained inline]",
            text,
        ]
        if failure_action:
            parts.append(failure_action)
        return "\n".join(parts)
    return _build_preview(
        text,
        head,
        tail,
        result["path"],
        source=source,
        success_action=success_action,
        failure_action=failure_action,
    )


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_PREVIEW_HEAD",
    "DEFAULT_PREVIEW_TAIL",
    "DEFAULT_ENABLED",
    "get_spill_config",
    "spill_if_oversized",
    "write_spill_file",
]
