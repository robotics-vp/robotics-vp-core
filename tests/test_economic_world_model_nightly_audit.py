from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "economic_world_model" / "nightly_audit.py"
    spec = importlib.util.spec_from_file_location("nightly_audit_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load nightly audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_latest_date_uses_most_recent_heading(monkeypatch) -> None:
    module = _load_audit_module()
    monkeypatch.setattr(
        module,
        "_read_text",
        lambda _: "\n".join(
            [
                "## 2026-03-26",
                "latest-first-entry",
                "## 2026-03-24",
                "older-entry",
            ]
        ),
    )
    assert module._progress_latest_date() == "2026-03-26"


def test_event_spine_spec_not_pending_when_code_and_docs_are_present(monkeypatch) -> None:
    module = _load_audit_module()

    architecture_path = module.REPO_ROOT / "docs/economic_world_model/architecture_gap_analysis.md"
    roadmap_path = module.REPO_ROOT / "docs/economic_world_model/roadmap.md"
    text_map = {
        architecture_path: "Event spine and Governance trace are documented.",
        roadmap_path: "Week 3 includes event spine + governance trace delivery.",
    }

    monkeypatch.setattr(module, "_exists", lambda _: True)
    monkeypatch.setattr(module, "_read_text", lambda path: text_map.get(path, ""))

    assert module._event_spine_spec_pending() is False


def test_next_task_falls_back_to_audit_only_when_candidates_are_complete(monkeypatch) -> None:
    module = _load_audit_module()
    monkeypatch.setattr(module, "_exists", lambda _: True)
    monkeypatch.setattr(module, "_search", lambda *_: True)
    monkeypatch.setattr(module, "_event_spine_spec_pending", lambda: False)
    monkeypatch.setattr(module, "_dataset_bridge_scaffold_pending", lambda: False)
    monkeypatch.setattr(module, "_future_training_evidence_pending", lambda: False)

    next_task = module._next_task([])
    assert next_task["id"] == "audit_only"


def test_next_task_picks_dataset_bridge_when_missing(monkeypatch) -> None:
    module = _load_audit_module()
    monkeypatch.setattr(module, "_exists", lambda _: True)
    monkeypatch.setattr(module, "_search", lambda *_: True)
    monkeypatch.setattr(module, "_event_spine_spec_pending", lambda: False)
    monkeypatch.setattr(module, "_dataset_bridge_scaffold_pending", lambda: True)
    monkeypatch.setattr(module, "_future_training_evidence_pending", lambda: False)

    next_task = module._next_task([])
    assert next_task["id"] == "dataset_bridge_scaffold"


def test_next_task_picks_future_training_evidence_when_shell_backlog_is_pending(monkeypatch) -> None:
    module = _load_audit_module()
    monkeypatch.setattr(module, "_exists", lambda _: True)
    monkeypatch.setattr(module, "_search", lambda *_: True)
    monkeypatch.setattr(module, "_event_spine_spec_pending", lambda: False)
    monkeypatch.setattr(module, "_dataset_bridge_scaffold_pending", lambda: False)
    monkeypatch.setattr(module, "_future_training_evidence_pending", lambda: True)

    next_task = module._next_task([])
    assert next_task["id"] == "future_training_evidence_wiring"


def test_next_task_prioritizes_agent_verify_failure_over_scaffolds() -> None:
    module = _load_audit_module()
    verification = [{"name": "agent_verify", "passed": False, "exit_code": 1}]

    next_task = module._next_task(verification)
    assert next_task["id"] == "agent_verify_regression"
    assert next_task["classification"] == "verification_hardening"


def test_next_task_prioritizes_generic_verification_failure_when_agent_verify_passes() -> None:
    module = _load_audit_module()
    verification = [{"name": "compileall", "passed": False, "exit_code": 1}]

    next_task = module._next_task(verification)
    assert next_task["id"] == "verification_regression"
    assert next_task["classification"] == "verification_hardening"
