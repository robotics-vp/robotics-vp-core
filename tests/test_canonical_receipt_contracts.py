from pathlib import Path

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
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_guardrail(root)

    assert payload["all_passed"] is True
    assert payload["issue_count"] == 0
