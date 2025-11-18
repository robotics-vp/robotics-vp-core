#!/usr/bin/env python3
"""
Semantic ↔ econ correlation report (analytics-only).

Reads ontology episodes/econ vectors plus optional semantic tags/advisories and
emits JSON/CSV summaries of econ outcomes grouped by semantic signals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_semantic_tags(path: Path) -> Dict[str, Dict]:
    """
    Returns episode_id -> enrichment dict with extracted fields of interest.
    Accepts Stage 2.4 SemanticEnrichmentProposal JSONL.
    """
    records = _read_jsonl(path)
    tags_by_ep: Dict[str, Dict] = {}
    for rec in records:
        ep_id = rec.get("episode_id")
        enrichment = rec.get("enrichment") or rec.get("enrichment", {})
        if not ep_id or enrichment is None:
            continue
        tags_by_ep[ep_id] = enrichment
    return tags_by_ep


def _load_advisories(path: Path) -> List[Dict]:
    return _read_jsonl(path)


def _econ_rows(store: OntologyStore, task_id: str = None) -> List[Dict]:
    episodes = store.list_episodes(task_id=task_id)
    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    rows: List[Dict] = []
    for ep in episodes:
        econ = econ_map.get(ep.episode_id)
        if not econ:
            continue
        comp = getattr(econ, "components", {}) or {}
        rows.append(
            {
                "episode_id": ep.episode_id,
                "task_id": ep.task_id,
                "mpl": float(econ.mpl_units_per_hour),
                "wage_parity": float(econ.wage_parity),
                "energy_cost": float(econ.energy_cost),
                "damage_cost": float(econ.damage_cost),
                "error_rate": float(comp.get("error_rate", 0.0)),
                "metadata": getattr(ep, "metadata", {}) or {},
            }
        )
    rows.sort(key=lambda r: r["episode_id"])
    return rows


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _summarize(rows: List[Dict]) -> Dict[str, float]:
    fields = ["mpl", "wage_parity", "energy_cost", "damage_cost", "error_rate"]
    return {f"mean_{f}": _mean([float(r.get(f, 0.0)) for r in rows]) for f in fields} | {"count": len(rows)}


def _label_rows_by_semantics(rows: List[Dict], tags_by_ep: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    labeled: Dict[str, List[Dict]] = {"all": rows}
    for r in rows:
        ep_id = r["episode_id"]
        enrichment = tags_by_ep.get(ep_id) or {}
        frag_tags = enrichment.get("fragility_tags") or []
        risk_tags = enrichment.get("risk_tags") or []
        nov_tags = enrichment.get("novelty_tags") or []
        supervision = enrichment.get("supervision_hints") or {}
        if frag_tags:
            labeled.setdefault("fragility", []).append(r)
        if any(t.get("severity") in {"high", "critical"} for t in risk_tags):
            labeled.setdefault("high_risk", []).append(r)
        if any(t.get("novelty_score", 0.0) >= 0.7 for t in nov_tags):
            labeled.setdefault("high_novelty", []).append(r)
        priority = supervision.get("priority_level")
        if priority:
            labeled.setdefault(f"priority_{str(priority).lower()}", []).append(r)
    return labeled


def _priority_from_advisory(advisory: Dict) -> str:
    safety = float(advisory.get("safety_emphasis", 0.0))
    if safety >= 0.75:
        return "CRITICAL"
    if safety >= 0.55:
        return "HIGH"
    if safety >= 0.35:
        return "MEDIUM"
    return "LOW"


def _label_rows_by_advisory(rows: List[Dict], advisories: List[Dict]) -> Dict[str, List[Dict]]:
    if not advisories:
        return {}
    rows_by_ep = {r["episode_id"]: r for r in rows}
    labeled: Dict[str, List[Dict]] = {}
    for adv in advisories:
        eps = adv.get("metadata", {}).get("frontier_eps") or []
        priority = _priority_from_advisory(adv)
        for ep in sorted(set(eps)):
            if ep in rows_by_ep:
                labeled.setdefault(priority, []).append(rows_by_ep[ep])
    return labeled


def _write_csv(rows: List[Tuple[str, Dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "mean_mpl", "mean_wage_parity", "mean_energy_cost", "mean_damage_cost", "mean_error_rate"])
        for label, stats in rows:
            writer.writerow(
                [
                    label,
                    stats.get("count", 0),
                    stats.get("mean_mpl", 0.0),
                    stats.get("mean_wage_parity", 0.0),
                    stats.get("mean_energy_cost", 0.0),
                    stats.get("mean_damage_cost", 0.0),
                    stats.get("mean_error_rate", 0.0),
                ]
            )


def run_report(
    ontology_root: str = "data/ontology",
    task_id: str = None,
    semantic_tags_path: str = None,
    advisories_path: str = None,
    output_dir: str = "results/semantic_econ",
):
    store = OntologyStore(root_dir=ontology_root)
    rows = _econ_rows(store, task_id=task_id)
    tags_by_ep = _load_semantic_tags(Path(semantic_tags_path)) if semantic_tags_path else {}
    advisories = _load_advisories(Path(advisories_path)) if advisories_path else []
    labeled = _label_rows_by_semantics(rows, tags_by_ep)
    advisory_labeled = _label_rows_by_advisory(rows, advisories)

    tag_stats = {k: _summarize(v) for k, v in sorted(labeled.items(), key=lambda kv: kv[0])}
    advisory_stats = {k: _summarize(v) for k, v in sorted(advisory_labeled.items(), key=lambda kv: kv[0])}
    summary = {
        "task_id": task_id,
        "total_episodes": len(rows),
        "tag_summaries": tag_stats,
        "advisory_summaries": advisory_stats,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "semantic_econ_correlations.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_rows = list(tag_stats.items())
    if advisory_stats:
        csv_rows.extend((f"advisory_{k}", v) for k, v in advisory_stats.items())
    _write_csv(csv_rows, out_dir / "semantic_econ_correlations.csv")

    # Human-readable summary
    print("[report_semantic_econ_correlations] totals:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic ↔ econ correlation report.")
    parser.add_argument("--ontology-root", default="data/ontology", help="Ontology store root directory.")
    parser.add_argument("--task-id", help="Task to summarize (optional; defaults to all).")
    parser.add_argument("--semantic-tags", help="Path to Stage 2.4 semantic tag enrichments (JSONL).")
    parser.add_argument("--advisories", help="Path to SemanticOrchestratorV2 advisories (JSONL).")
    parser.add_argument("--output-dir", default="results/semantic_econ", help="Report output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_report(
        ontology_root=args.ontology_root,
        task_id=args.task_id,
        semantic_tags_path=args.semantic_tags,
        advisories_path=args.advisories,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
