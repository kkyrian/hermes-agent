from pathlib import Path


def test_claude_instruction_entrypoint_is_portable_regular_file():
    root = Path(__file__).resolve().parents[2]
    entrypoint = root / "CLAUDE.md"

    assert entrypoint.is_file()
    assert not entrypoint.is_symlink()
    assert "Read and follow [AGENTS.md](AGENTS.md) completely" in entrypoint.read_text(
        encoding="utf-8"
    )
