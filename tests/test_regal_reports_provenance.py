import numpy as np

from src.provenance import extract_stamped_metadata, stamp_regal_decision_metadata
from src.regal.base import RegalDecision, RegalReport


def test_regal_report_provenance_stamped_roundtrip(tmp_path):
    report = RegalReport(
        node_id="regal_test",
        decision=RegalDecision.REROUTE,
        reason_codes=["demo_reason"],
        details={"x": 1},
    )
    payload = {"dummy": np.array([0.0], dtype=np.float32)}
    payload = stamp_regal_decision_metadata(payload, regal_decision_metadata=report.to_dict())
    out = tmp_path / "regal_provenance.npz"
    np.savez_compressed(out, **payload)

    loaded = dict(np.load(out, allow_pickle=False))
    restored = extract_stamped_metadata(loaded)
    assert restored["regal_decision"]["node_id"] == "regal_test"
    assert restored["regal_decision"]["decision"] == RegalDecision.REROUTE.value
