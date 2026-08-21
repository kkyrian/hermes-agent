from agent.prompt_builder import build_context_files_prompt


def test_repository_instruction_discovery_prefers_agents_over_claude(tmp_path):
    (tmp_path / "AGENTS.md").write_text(
        "synthetic agents marker", encoding="utf-8"
    )
    (tmp_path / "CLAUDE.md").write_text(
        "synthetic claude marker", encoding="utf-8"
    )
    forwarded = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)

    assert "## AGENTS.md" in forwarded
    assert "synthetic agents marker" in forwarded
    assert "synthetic claude marker" not in forwarded
