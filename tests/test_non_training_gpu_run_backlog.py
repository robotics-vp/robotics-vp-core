from __future__ import annotations

import json
from pathlib import Path

from src.orchestrator.non_training_gpu_run_backlog import (
    evaluate_non_training_gpu_run_backlog,
    load_non_training_gpu_run_backlog,
)


def test_non_training_gpu_run_backlog_loads_and_evaluates(tmp_path: Path) -> None:
    backlog_path = tmp_path / "gpu_backlog.json"
    backlog_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "backlog": [
                    {
                        "loop_run_id": "gpu_smoke",
                        "title": "gpu smoke",
                        "command": "python3 scripts/local_isaac_smoke.py",
                        "required_capabilities": {
                            "gpu_available": True,
                            "isaaclab_backend_module_available": True,
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    items = load_non_training_gpu_run_backlog(backlog_path)
    assessments = evaluate_non_training_gpu_run_backlog(
        backlog_path=backlog_path,
        host_capabilities={
            "gpu_available": True,
            "isaaclab_backend_module_available": True,
        },
    )

    assert len(items) == 1
    assert items[0].loop_run_id == "gpu_smoke"
    assert len(assessments) == 1
    assert assessments[0].ready is True
