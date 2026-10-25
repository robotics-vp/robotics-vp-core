from pathlib import Path

import pytest

from scripts.check_canonical_receipt_contracts import (
    find_provider_truth_issues,
    find_receipt_contract_issues,
    run_guardrail,
)


def test_receipt_guardrail_flags_missing_required_fields(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "orchestrator"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "bad_receipt.py").write_text(
        'PAYLOAD = {"receipt_kind": "bad_receipt_v1"}\n',
        encoding="utf-8",
    )

    issues = find_receipt_contract_issues(root, relative_dirs=("src/orchestrator",))

    assert len(issues) == 1
    assert issues[0]["kind"] == "receipt_contract_missing_fields"
    assert "authority_class" in issues[0]["detail"]


def test_provider_truth_guardrail_flags_missing_canonicalization(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "vla"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "bad_provider_truth.py").write_text(
        'payload = {"provider_truth": {"backend_selected": "real"}}\n',
        encoding="utf-8",
    )

    issues = find_provider_truth_issues(root, relative_dirs=("src/vla",))

    assert len(issues) == 1
    assert issues[0]["kind"] == "provider_truth_missing_canonicalization"


def test_provider_truth_guardrail_ignores_descriptive_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "vision"
    target_dir.mkdir(parents=True)
    (target_dir / "provider_metadata.py").write_text(
        'metadata = {"provider_truth_class": "stub_smoke_only", '
        '"provider_truth_classes": ["blocked_provider"], '
        '"provider_truth_owner": "sim_synth_physics_wm", '
        '"provider_truth_claim": "no_provider_bringup_claimed"}\n',
        encoding="utf-8",
    )

    assert find_provider_truth_issues(root, relative_dirs=("src/vision",)) == []


@pytest.mark.parametrize("field_name", ["provider_truth", "scene_tracks_provider_truth"])
def test_provider_truth_metadata_does_not_exempt_uncanonical_payloads(
    tmp_path: Path, field_name: str
) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "vision"
    target_dir.mkdir(parents=True)
    (target_dir / "mixed_payload.py").write_text(
        'metadata = {"provider_truth_class": "stub_smoke_only"}\n'
        f'payload = {{"{field_name}": {{"backend_selected": "real"}}}}\n',
        encoding="utf-8",
    )

    issues = find_provider_truth_issues(root, relative_dirs=("src/vision",))

    assert len(issues) == 1
    assert issues[0]["kind"] == "provider_truth_missing_canonicalization"


def test_provider_truth_guardrail_accepts_canonical_helper_without_inline_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "vision"
    target_dir.mkdir(parents=True)
    (target_dir / "canonical_payload.py").write_text(
        "from src.evidence.provider_truth import build_external_provider_truth\n"
        'payload = {"provider_truth": build_external_provider_truth('
        'provider_id="x", provider_kind="y")}\n',
        encoding="utf-8",
    )

    assert find_provider_truth_issues(root, relative_dirs=("src/vision",)) == []


def test_guardrail_passes_on_compliant_contracts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target_dir = root / "src" / "evidence"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "good_contract.py").write_text(
        "\n".join(
            [
                'from src.evidence.provider_truth import build_external_provider_truth',
                'PAYLOAD = {"receipt_kind": "ok_v1", "authority_class": "canonical_metadata", "decision_scope": "x", "reward_math_mutation": False}',
                'provider_truth = build_external_provider_truth(provider_id="x", provider_kind="y")',
                'metadata = {"provider_truth_class": "stub_smoke_only"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_guardrail(root)

    assert payload["all_passed"] is True
    assert payload["issue_count"] == 0
