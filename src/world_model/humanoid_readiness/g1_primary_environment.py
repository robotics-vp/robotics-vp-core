"""Canonical Unitree G1 primary-environment doctrine and hygiene sweep.

This module is structural plumbing only. It makes the repo-wide default target
explicit as a Unitree G1 bipedal whole-body environment while preserving
workcell, drawer/vase, and dishwashing surfaces as fixed-base curriculum or
regression inputs. It does not run sim, hardware, training, providers, or
promotion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe

G1_PRIMARY_ENV_DOCTRINE_VERSION = "g1_primary_environment_doctrine_v1"
G1_PRIMARY_ENV_HYGIENE_REPORT_VERSION = "g1_primary_env_hygiene_report_v1"
G1_PRIMARY_ENV_HYGIENE_RECEIPT_VERSION = "g1_primary_env_hygiene_receipt_v1"

PRIMARY_ENV_TYPE = "unitree_g1"
PRIMARY_ENV_ID = "bipedal_whole_body_unitree_g1"
PRIMARY_TASK_ID = "humanoid_wbt_g1"
PRIMARY_ROBOT_FAMILY = "unitree_g1"
PRIMARY_HARDWARE_CLASS = "unitree_g1_r1_class"
PRIMARY_EMBODIMENT_ID = "unitree_g1_shadow"
PRIMARY_POSTURE_TAG = "bipedal_whole_body"
FALLBACK_POSTURE_TAG = "stable_base_mobile_manipulator"
CURRICULUM_POSTURE_TAG = "fixed_base_tabletop"

LEGACY_CURRICULUM_ENVS = {
    "dishwashing": "fixed_base_tabletop curriculum/regression source",
    "dishwashing_env": "fixed_base_tabletop curriculum/regression source",
    "dishwashing_online_sac": "fixed_base_tabletop SAC plumbing source",
    "drawer_vase": "fixed_base_tabletop manipulation curriculum source",
    "drawer_vase_env": "fixed_base_tabletop manipulation curriculum source",
    "workcell": "fixed_base_tabletop workcell curriculum/replay source",
    "workcell_env": "fixed_base_tabletop workcell curriculum/replay source",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SCAN_ROOTS = (
    ".github",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests",
)
ROOT_SCAN_FILES = ("AGENTS.md", "train_sac.py", "pyproject.toml")
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "data",
    "results",
}

PRIMARY_CLAIM_RE = re.compile(
    r"\b(primary|paramount|canonical|blessed)\b|"
    r"\bdefault\b(?!\.(?:cfg|ini|json|md|py|sh|toml|txt|ya?ml)\b)",
    re.IGNORECASE,
)
LEGACY_ENV_RE = re.compile(
    r"\b(workcell|dishwashing|drawer[_/-]?vase|tabletop)\b", re.IGNORECASE
)
BOUNDARY_RE = re.compile(
    r"\b(curriculum|legacy|regression|proxy|source|fixed_base_tabletop|"
    r"not primary|not the primary|fallback|historical|deprecated|smoke|"
    r"testbed|allowed|only|G1-primary|G1 primary|g1-primary|g1 primary)\b",
    re.IGNORECASE,
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_mapping(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_mapping(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class G1PrimaryEnvironmentDoctrine:
    """Repo-wide primary target and allowed lower-posture roles."""

    env_type: str = PRIMARY_ENV_TYPE
    primary_env_id: str = PRIMARY_ENV_ID
    primary_task_id: str = PRIMARY_TASK_ID
    primary_robot_family: str = PRIMARY_ROBOT_FAMILY
    primary_hardware_class: str = PRIMARY_HARDWARE_CLASS
    primary_embodiment_id: str = PRIMARY_EMBODIMENT_ID
    primary_posture_tag: str = PRIMARY_POSTURE_TAG
    fallback_posture_tag: str = FALLBACK_POSTURE_TAG
    curriculum_posture_tag: str = CURRICULUM_POSTURE_TAG
    legacy_curriculum_envs: dict[str, str] = field(
        default_factory=lambda: dict(LEGACY_CURRICULUM_ENVS)
    )
    denied_claims: list[str] = field(
        default_factory=lambda: [
            "unitree_hardware_truth",
            "unitree_sim_runtime_truth",
            "live_policy_control",
            "trained_whole_body_policy",
            "promotion_eligible",
        ]
    )
    version: str = G1_PRIMARY_ENV_DOCTRINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "env_type": self.env_type,
            "primary_env_id": self.primary_env_id,
            "primary_task_id": self.primary_task_id,
            "primary_robot_family": self.primary_robot_family,
            "primary_hardware_class": self.primary_hardware_class,
            "primary_embodiment_id": self.primary_embodiment_id,
            "primary_posture_tag": self.primary_posture_tag,
            "fallback_posture_tag": self.fallback_posture_tag,
            "curriculum_posture_tag": self.curriculum_posture_tag,
            "legacy_curriculum_envs": dict(self.legacy_curriculum_envs),
            "denied_claims": list(self.denied_claims),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "G1PrimaryEnvironmentDoctrine":
        return cls(
            env_type=str(payload.get("env_type", PRIMARY_ENV_TYPE)),
            primary_env_id=str(payload.get("primary_env_id", PRIMARY_ENV_ID)),
            primary_task_id=str(payload.get("primary_task_id", PRIMARY_TASK_ID)),
            primary_robot_family=str(
                payload.get("primary_robot_family", PRIMARY_ROBOT_FAMILY)
            ),
            primary_hardware_class=str(
                payload.get("primary_hardware_class", PRIMARY_HARDWARE_CLASS)
            ),
            primary_embodiment_id=str(
                payload.get("primary_embodiment_id", PRIMARY_EMBODIMENT_ID)
            ),
            primary_posture_tag=str(
                payload.get("primary_posture_tag", PRIMARY_POSTURE_TAG)
            ),
            fallback_posture_tag=str(
                payload.get("fallback_posture_tag", FALLBACK_POSTURE_TAG)
            ),
            curriculum_posture_tag=str(
                payload.get("curriculum_posture_tag", CURRICULUM_POSTURE_TAG)
            ),
            legacy_curriculum_envs={
                str(key): str(value)
                for key, value in dict(
                    payload.get("legacy_curriculum_envs")
                    or LEGACY_CURRICULUM_ENVS
                ).items()
            },
            denied_claims=_strings(payload.get("denied_claims")),
            version=str(payload.get("version", G1_PRIMARY_ENV_DOCTRINE_VERSION)),
        )


@dataclass(frozen=True)
class G1PrimaryEnvHygieneReceipt:
    receipt_id: str
    check_key: str
    status: str
    passed: bool
    severity: str = "blocking"
    measured_value: Any = None
    blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = G1_PRIMARY_ENV_HYGIENE_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "check_key": self.check_key,
            "status": self.status,
            "passed": bool(self.passed),
            "severity": self.severity,
            "measured_value": to_json_safe(self.measured_value),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class G1PrimaryEnvHygieneReport:
    report_id: str
    status: str
    scanned_file_count: int
    primary_doctrine_present: bool
    required_surface_count: int
    missing_required_surface_count: int
    legacy_primary_claim_count: int
    advisory_legacy_reference_count: int
    blocking_issue_count: int
    advisory_issue_count: int
    receipt_count: int
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = G1_PRIMARY_ENV_HYGIENE_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "status": self.status,
            "scanned_file_count": int(self.scanned_file_count),
            "primary_doctrine_present": bool(self.primary_doctrine_present),
            "required_surface_count": int(self.required_surface_count),
            "missing_required_surface_count": int(
                self.missing_required_surface_count
            ),
            "legacy_primary_claim_count": int(self.legacy_primary_claim_count),
            "advisory_legacy_reference_count": int(
                self.advisory_legacy_reference_count
            ),
            "blocking_issue_count": int(self.blocking_issue_count),
            "advisory_issue_count": int(self.advisory_issue_count),
            "receipt_count": int(self.receipt_count),
            "output_paths": _mapping(self.output_paths),
            "metadata": _mapping(self.metadata),
        }


def default_g1_primary_environment_doctrine() -> G1PrimaryEnvironmentDoctrine:
    return G1PrimaryEnvironmentDoctrine()


def primary_env_metadata(
    *,
    primary_env_id: str = PRIMARY_ENV_ID,
    primary_task_id: str = PRIMARY_TASK_ID,
    robot_family: str = PRIMARY_ROBOT_FAMILY,
    posture_tag: str = PRIMARY_POSTURE_TAG,
    embodiment_id: str = PRIMARY_EMBODIMENT_ID,
    source_curriculum_env: str = "",
) -> dict[str, Any]:
    """Return JSON-safe metadata for a G1-primary training or replay surface."""

    payload = {
        "env_type": PRIMARY_ENV_TYPE,
        "primary_env_id": primary_env_id,
        "primary_task_id": primary_task_id,
        "primary_robot_family": robot_family,
        "primary_embodiment_id": embodiment_id,
        "primary_posture_tag": posture_tag,
        "primary_hardware_class": PRIMARY_HARDWARE_CLASS,
        "fallback_posture_tag": FALLBACK_POSTURE_TAG,
        "curriculum_posture_tag": CURRICULUM_POSTURE_TAG,
        "source_curriculum_env": source_curriculum_env,
        "unitree_hardware_truth": False,
        "unitree_sim_runtime_truth": False,
        "live_policy_control": False,
        "promotion_eligible": False,
        "authority_class": "g1_primary_env_shadow_contract",
    }
    if source_curriculum_env:
        payload["source_curriculum"] = curriculum_proxy_metadata(
            source_curriculum_env
        )
    return payload


def curriculum_proxy_metadata(source_env: str) -> dict[str, Any]:
    """Return the explicit boundary for a non-G1 curriculum source."""

    key = str(source_env or "unknown").strip()
    return {
        "source_env": key,
        "posture_tag": CURRICULUM_POSTURE_TAG,
        "role": LEGACY_CURRICULUM_ENVS.get(
            key, "fixed_base_tabletop curriculum/regression source"
        ),
        "promotion_limit": "cannot_close_g1_r1_whole_body_readiness",
        "may_feed_primary_target": True,
        "authority_class": "curriculum_source_only",
    }


def classify_env_posture(env_name: str) -> str:
    text = str(env_name or "").lower()
    if "g1" in text or "bipedal" in text or "humanoid_wbt" in text:
        return PRIMARY_POSTURE_TAG
    if "stable_base" in text or "mobile_manipulator" in text:
        return FALLBACK_POSTURE_TAG
    if text in LEGACY_CURRICULUM_ENVS or any(
        token in text for token in ("workcell", "dishwashing", "drawer_vase")
    ):
        return CURRICULUM_POSTURE_TAG
    return "unknown"


def _receipt(
    *,
    check_key: str,
    passed: bool,
    measured_value: Any = None,
    severity: str = "blocking",
    blocker: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> G1PrimaryEnvHygieneReceipt:
    return G1PrimaryEnvHygieneReceipt(
        receipt_id=_stable_id(
            "g1_primary_env_hygiene",
            {
                "check_key": check_key,
                "measured_value": measured_value,
                "metadata": dict(metadata or {}),
            },
        ),
        check_key=check_key,
        status="ok" if passed else "blocked",
        passed=passed,
        severity=severity,
        measured_value=measured_value,
        blockers=[] if passed else [blocker or check_key],
        metadata=dict(metadata or {}),
    )


def _iter_scan_files(repo_root: Path) -> Iterable[Path]:
    for file_name in ROOT_SCAN_FILES:
        path = repo_root / file_name
        if path.exists():
            yield path
    for rel_root in SCAN_ROOTS:
        root = repo_root / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_PARTS for part in path.parts):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def _is_legacy_primary_claim(line: str) -> bool:
    return (
        bool(PRIMARY_CLAIM_RE.search(line))
        and bool(LEGACY_ENV_RE.search(line))
        and not BOUNDARY_RE.search(line)
    )


def _line_refs(repo_root: Path) -> tuple[list[dict[str, Any]], int, int]:
    risky: list[dict[str, Any]] = []
    advisory_count = 0
    scanned_count = 0
    for path in sorted(set(_iter_scan_files(repo_root))):
        scanned_count += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if _is_legacy_primary_claim(line):
                risky.append(
                    {
                        "path": str(path.relative_to(repo_root)),
                        "line": idx,
                        "text": line.strip()[:240],
                    }
                )
            elif LEGACY_ENV_RE.search(line) and BOUNDARY_RE.search(line):
                advisory_count += 1
    return risky, advisory_count, scanned_count


def run_g1_primary_env_hygiene(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Sweep repo text for old primary-env claims and required G1 surfaces."""

    root = Path(repo_root)
    out = Path(output_dir)
    required_surfaces = [
        "configs/humanoid/g1_primary_env.yaml",
        "docs/economic_world_model/g1_primary_environment.md",
        "scripts/economic_world_model/check_g1_primary_env_hygiene.py",
        "scripts/runpod/prepare_launch_manifest.py",
        "src/world_model/humanoid_readiness/g1_primary_environment.py",
        "src/runpod/launch_profiles.py",
        "configs/runpod/examples/train_sac_manifest.json",
        "configs/runpod/examples/provider_bringup_manifest.json",
    ]
    missing_surfaces = [
        rel_path for rel_path in required_surfaces if not (root / rel_path).exists()
    ]
    risky_lines, advisory_legacy_count, scanned_file_count = _line_refs(root)
    doctrine = default_g1_primary_environment_doctrine()
    primary_doctrine_present = (
        doctrine.primary_env_id == PRIMARY_ENV_ID
        and doctrine.primary_posture_tag == PRIMARY_POSTURE_TAG
        and doctrine.legacy_curriculum_envs.get("dishwashing")
        is not None
    )

    receipts = [
        _receipt(
            check_key="primary_doctrine_present",
            passed=primary_doctrine_present,
            measured_value=doctrine.to_dict(),
            blocker="g1_primary_doctrine_missing_or_invalid",
        ),
        _receipt(
            check_key="required_surfaces_present",
            passed=not missing_surfaces,
            measured_value=missing_surfaces,
            blocker="g1_primary_required_surfaces_missing",
        ),
        _receipt(
            check_key="legacy_env_not_claimed_primary",
            passed=not risky_lines,
            measured_value=risky_lines,
            blocker="legacy_env_primary_or_default_claim_without_curriculum_boundary",
        ),
    ]
    blocking_issue_count = sum(
        1
        for receipt in receipts
        if not receipt.passed and receipt.severity == "blocking"
    )
    advisory_issue_count = sum(
        1
        for receipt in receipts
        if not receipt.passed and receipt.severity != "blocking"
    )
    output_paths = {
        "receipts_path": str(out / "g1_primary_env_hygiene_receipts_v1.jsonl"),
        "report_path": str(out / "g1_primary_env_hygiene_report_v1.json"),
    }
    report = G1PrimaryEnvHygieneReport(
        report_id=_stable_id(
            "g1_primary_env_hygiene_report",
            {
                "required_surfaces": required_surfaces,
                "legacy_primary_claim_count": len(risky_lines),
                "blocking_issue_count": blocking_issue_count,
            },
        ),
        status="ok_g1_primary_env_hygiene_passed"
        if blocking_issue_count == 0
        else "blocked_g1_primary_env_hygiene_failed",
        scanned_file_count=scanned_file_count,
        primary_doctrine_present=primary_doctrine_present,
        required_surface_count=len(required_surfaces),
        missing_required_surface_count=len(missing_surfaces),
        legacy_primary_claim_count=len(risky_lines),
        advisory_legacy_reference_count=advisory_legacy_count,
        blocking_issue_count=blocking_issue_count,
        advisory_issue_count=advisory_issue_count,
        receipt_count=len(receipts),
        output_paths=output_paths,
        metadata={
            "doctrine": doctrine.to_dict(),
            "required_surfaces": required_surfaces,
            "legacy_primary_claim_refs": risky_lines,
        },
    )
    _write_jsonl(
        Path(output_paths["receipts_path"]),
        [receipt.to_dict() for receipt in receipts],
    )
    _write_json(Path(output_paths["report_path"]), report.to_dict())
    return report.to_dict()


__all__ = [
    "CURRICULUM_POSTURE_TAG",
    "FALLBACK_POSTURE_TAG",
    "G1PrimaryEnvHygieneReceipt",
    "G1PrimaryEnvHygieneReport",
    "G1PrimaryEnvironmentDoctrine",
    "G1_PRIMARY_ENV_DOCTRINE_VERSION",
    "G1_PRIMARY_ENV_HYGIENE_REPORT_VERSION",
    "G1_PRIMARY_ENV_HYGIENE_RECEIPT_VERSION",
    "LEGACY_CURRICULUM_ENVS",
    "PRIMARY_EMBODIMENT_ID",
    "PRIMARY_ENV_ID",
    "PRIMARY_ENV_TYPE",
    "PRIMARY_HARDWARE_CLASS",
    "PRIMARY_POSTURE_TAG",
    "PRIMARY_ROBOT_FAMILY",
    "PRIMARY_TASK_ID",
    "classify_env_posture",
    "curriculum_proxy_metadata",
    "default_g1_primary_environment_doctrine",
    "primary_env_metadata",
    "run_g1_primary_env_hygiene",
]
