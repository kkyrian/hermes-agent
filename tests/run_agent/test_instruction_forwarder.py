from pathlib import Path

from agent.prompt_builder import build_context_files_prompt


def test_repository_instruction_discovery_forwards_canonical_agents_contract():
    root = Path(__file__).resolve().parents[2]
    forwarded = build_context_files_prompt(cwd=str(root), skip_soul=True)

    assert "## AGENTS.md" in forwarded
    assert "## KK public fork maintenance" in forwarded
    assert "`kk/handler-runtime` is the single durable KK patch branch" in forwarded
    assert "Claude entry point" not in forwarded
