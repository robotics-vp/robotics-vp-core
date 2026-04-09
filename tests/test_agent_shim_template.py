from pathlib import Path


def test_claude_shim_matches_canonical_template() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shim_template = repo_root / "scripts" / "agent" / "claude_shim_template.md"
    claude_md = repo_root / "CLAUDE.md"

    assert shim_template.exists(), "Expected canonical CLAUDE shim template is missing"
    assert claude_md.exists(), "CLAUDE.md must exist at repo root"
    assert claude_md.read_text().rstrip("\n") == shim_template.read_text().rstrip("\n")


def test_claude_shim_template_includes_copilot_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shim_template = repo_root / "scripts" / "agent" / "claude_shim_template.md"
    content = shim_template.read_text()
    assert "@.agent/claude_copilot.md" in content
