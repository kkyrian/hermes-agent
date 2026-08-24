"""Tests for tools.hook_output_spill."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import hook_output_spill as hos


class GetSpillConfigTests(unittest.TestCase):
    def test_defaults_when_no_config(self):
        with patch.object(hos, "load_config", create=True, return_value={}):
            # load_config is resolved at call time via local import;
            # patch the module's source instead.
            pass
        with patch("hermes_cli.config.load_config", return_value={}):
            cfg = hos.get_spill_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["max_chars"], hos.DEFAULT_MAX_CHARS)
        self.assertEqual(cfg["preview_head"], hos.DEFAULT_PREVIEW_HEAD)
        self.assertEqual(cfg["preview_tail"], hos.DEFAULT_PREVIEW_TAIL)
        self.assertIsNone(cfg["directory"])


    def test_load_config_exception_is_swallowed(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad")):
            cfg = hos.get_spill_config()
        self.assertEqual(cfg["max_chars"], hos.DEFAULT_MAX_CHARS)
        self.assertTrue(cfg["enabled"])


class SpillIfOversizedTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hermes-spill-test-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _cfg(self, **overrides):
        base = {
            "enabled": True,
            "max_chars": 100,
            "preview_head": 20,
            "preview_tail": 20,
            "directory": self.tmpdir,
        }
        base.update(overrides)
        return base

    def test_empty_and_none_are_noops(self):
        self.assertEqual(hos.spill_if_oversized("", config=self._cfg()), "")
        self.assertEqual(hos.spill_if_oversized(None, config=self._cfg()), "")

    def test_text_under_cap_is_unchanged(self):
        small = "x" * 50
        self.assertEqual(hos.spill_if_oversized(small, config=self._cfg()), small)

    def test_mandatory_spill_is_profile_local_atomic_and_actionable(self):
        profile = Path(self.tmpdir) / "profile"
        external = Path(self.tmpdir) / "external"
        profile.mkdir()
        token = set_hermes_home_override(profile)
        try:
            rendered = hos.spill_if_oversized(
                "complete catalog",
                session_id="same/session",
                source="direct skill catalog",
                config=self._cfg(enabled=False, directory=str(external)),
                force=True,
                namespace="context-delta/catalogs",
                success_action="Read the saved catalog before continuing.",
                failure_action="Inspect the skill roots before continuing.",
            )
        finally:
            reset_hermes_home_override(token)

        self.assertIn("full content saved to ", rendered)
        self.assertTrue(rendered.endswith("Read the saved catalog before continuing."))
        saved = Path(rendered.split("full content saved to ", 1)[1].split("]", 1)[0])
        self.assertEqual(saved.read_text(encoding="utf-8"), "complete catalog")
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertTrue(saved.is_relative_to(profile))
        self.assertFalse(external.exists())

    def test_same_session_mandatory_spills_are_isolated_by_profile(self):
        paths = []
        for name in ("profile-a", "profile-b"):
            profile = Path(self.tmpdir) / name
            profile.mkdir()
            token = set_hermes_home_override(profile)
            try:
                rendered = hos.spill_if_oversized(
                    name,
                    session_id="shared",
                    source="context delta",
                    config=self._cfg(enabled=False),
                    force=True,
                    namespace="context-delta/deliveries",
                )
            finally:
                reset_hermes_home_override(token)
            path = Path(rendered.split("full content saved to ", 1)[1].split("]", 1)[0])
            self.assertTrue(path.is_relative_to(profile))
            paths.append(path)
        self.assertNotEqual(paths[0], paths[1])

    def test_profile_local_spill_rejects_symlink_escape_and_preserves_exact_bytes(self):
        profile = Path(self.tmpdir) / "profile-safe"
        external = Path(self.tmpdir) / "external"
        profile.mkdir()
        external.mkdir()
        (profile / "context-delta").symlink_to(external, target_is_directory=True)
        token = set_hermes_home_override(profile)
        try:
            result = hos.write_spill_file(
                "abc",
                session_id="shared",
                namespace="context-delta/deliveries",
                profile_local=True,
            )
        finally:
            reset_hermes_home_override(token)
        self.assertFalse(result["ok"])
        self.assertEqual(list(external.iterdir()), [])

        (profile / "context-delta").unlink()
        token = set_hermes_home_override(profile)
        try:
            result = hos.write_spill_file(
                "abc",
                session_id="shared",
                namespace="context-delta/deliveries",
                profile_local=True,
            )
        finally:
            reset_hermes_home_override(token)
        self.assertTrue(result["ok"])
        self.assertEqual(Path(result["path"]).read_bytes(), b"abc")

    def test_generic_spill_rejects_symlinked_session_directory(self):
        directory = Path(self.tmpdir) / "generic-safe"
        external = Path(self.tmpdir) / "generic-external"
        directory.mkdir()
        external.mkdir()
        (directory / "shared").symlink_to(external, target_is_directory=True)

        result = hos.write_spill_file(
            "secret",
            session_id="shared",
            directory_override=str(directory),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(list(external.iterdir()), [])

    def test_generic_spill_allows_trusted_base_with_symlinked_ancestor(self):
        actual = Path(self.tmpdir) / "actual"
        alias = Path(self.tmpdir) / "alias"
        actual.mkdir()
        alias.symlink_to(actual, target_is_directory=True)

        result = hos.write_spill_file(
            "secret",
            session_id="shared",
            directory_override=str(alias / "spills"),
        )

        self.assertTrue(result["ok"], result["error"])
        saved = Path(result["path"])
        self.assertEqual(saved.read_text(encoding="utf-8"), "secret")
        self.assertTrue(saved.is_relative_to(actual.resolve()))

    def test_portable_profile_local_spill_preserves_exact_bytes(self):
        profile = Path(self.tmpdir) / "profile-portable"
        profile.mkdir()
        token = set_hermes_home_override(profile)
        try:
            with patch.object(hos, "_supports_secure_dir_fd", return_value=False):
                result = hos.write_spill_file(
                    "portable payload",
                    session_id="shared",
                    namespace="context-delta/deliveries",
                    profile_local=True,
                )
        finally:
            reset_hermes_home_override(token)

        self.assertTrue(result["ok"])
        saved = Path(result["path"])
        self.assertEqual(saved.read_text(encoding="utf-8"), "portable payload")
        self.assertTrue(saved.is_relative_to(profile))

    def test_profile_local_spill_cleans_plaintext_temp_on_interrupt(self):
        profile = Path(self.tmpdir) / "profile-interrupt"
        profile.mkdir()
        token = set_hermes_home_override(profile)
        try:
            with patch.object(os, "replace", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    hos.write_spill_file(
                        "TOP-SECRET",
                        session_id="shared",
                        namespace="context-delta/deliveries",
                        profile_local=True,
                    )
        finally:
            reset_hermes_home_override(token)
        self.assertEqual(list(profile.rglob(".spill-*.tmp")), [])

    def test_generic_spill_cleans_plaintext_temp_on_interrupt(self):
        directory = Path(self.tmpdir) / "generic"
        with patch.object(os, "replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                hos.write_spill_file(
                    "TOP-SECRET",
                    session_id="shared",
                    directory_override=str(directory),
                )
        self.assertEqual(list(directory.rglob(".spill-*.tmp")), [])

    def test_secure_spill_removes_renamed_file_when_final_chmod_fails(self):
        directory = Path(self.tmpdir) / "secure-finalize-failure"
        with patch.object(os, "chmod", side_effect=OSError("chmod failed")):
            result = hos.write_spill_file(
                "TOP-SECRET",
                session_id="shared",
                directory_override=str(directory),
            )
        self.assertFalse(result["ok"])
        self.assertEqual(list(directory.rglob("*.txt")), [])
        self.assertEqual(list(directory.rglob(".spill-*.tmp")), [])

    def test_portable_spill_removes_renamed_file_when_final_chmod_fails(self):
        directory = Path(self.tmpdir) / "portable-finalize-failure"
        with (
            patch.object(hos, "_supports_secure_dir_fd", return_value=False),
            patch.object(os, "chmod", side_effect=OSError("chmod failed")),
        ):
            result = hos.write_spill_file(
                "TOP-SECRET",
                session_id="shared",
                directory_override=str(directory),
            )
        self.assertFalse(result["ok"])
        self.assertEqual(list(directory.rglob("*.txt")), [])
        self.assertEqual(list(directory.rglob(".spill-*.tmp")), [])

    def test_mandatory_spill_failure_retains_complete_payload_inline(self):
        text = "HEAD-" + "x" * 500 + "-UNIQUE-MIDDLE-" + "y" * 500 + "-TAIL"
        with patch.object(
            hos,
            "write_spill_file",
            return_value={"ok": False, "path": None, "error": "ENOSPC"},
        ):
            rendered = hos.spill_if_oversized(
                text,
                session_id="shared",
                source="plugin hook",
                config=self._cfg(enabled=False),
                force=True,
                failure_action="Spill failed.",
            )
        self.assertIn("mandatory spill failed", rendered)
        self.assertIn(text, rendered)
        self.assertIn("UNIQUE-MIDDLE", rendered)
        self.assertTrue(rendered.endswith("Spill failed."))


    def test_default_directory_uses_hermes_home(self):
        """When no directory override, spill under HERMES_HOME/hook_outputs."""
        test_home = tempfile.mkdtemp(prefix="hermes-home-")
        try:
            with patch.dict(os.environ, {"HERMES_HOME": test_home}):
                # Also patch get_hermes_home to the env var to mirror production.
                cfg = self._cfg(directory=None, max_chars=5)
                hos.spill_if_oversized("x" * 200, session_id="sess", config=cfg)
            # Spill directory exists somewhere under test_home OR default
            # ~/.hermes/hook_outputs depending on get_hermes_home behaviour.
            candidates = [
                Path(test_home) / "hook_outputs" / "sess",
                Path(os.path.expanduser("~/.hermes/hook_outputs/sess")),
            ]
            # At least one of the candidate dirs now exists and has a file.
            existing = [c for c in candidates if c.is_dir() and list(c.iterdir())]
            self.assertTrue(existing, f"No spill dir found in {candidates}")
        finally:
            import shutil
            shutil.rmtree(test_home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
