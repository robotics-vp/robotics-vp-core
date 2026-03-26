import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.train_meta_transformer_synthetic import _run_training
from src.orchestrator.meta_transformer import MetaTransformer
from src.orchestrator.meta_transformer_training import (
    generate_meta_transformer_dataset,
    save_meta_transformer_dataset,
)


def _write_runtime_export(root: Path, sample_count: int = 8) -> Path:
    export_dir = root / "runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "meta_transformer_runtime_dataset.json"
    summary_path = export_dir / "semantic_runtime_learning_summary.json"
    save_meta_transformer_dataset(generate_meta_transformer_dataset(sample_count), str(dataset_path))
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_runtime_learning_summary_v1",
                "total_rows": sample_count,
                "bounded_ready_count": sample_count,
                "semantic_grounded_count": sample_count,
                "route_success_count": sample_count // 2,
                "authority_success_count": sample_count // 2,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return export_dir


def _train_runtime_package(tmp_path: Path) -> str:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        synthetic_samples=0,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_name="meta_runtime",
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        max_semantic_tokens=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        val_fraction=0.25,
        seed=17,
        skip_regal_runner=True,
    )
    return _run_training(args, runner=None)["runtime_package"]


def test_meta_transformer_auto_mode_consumes_runtime_package(tmp_path: Path) -> None:
    runtime_package = _train_runtime_package(tmp_path)
    transformer = MetaTransformer(
        d_shared=24,
        helper_package_path=runtime_package,
        helper_mode="auto",
    )

    output = transformer.forward(
        dino_features=np.zeros(256, dtype=np.float32),
        vla_features=np.zeros(128, dtype=np.float32),
    )

    helper = output.metadata["learned_helper"]
    assert helper["status"] == "loaded"
    assert helper["promotion_stage"] == "shadow_candidate"
    assert helper["benchmark_gate_ready"] is False
    assert helper["helper_weight"] == pytest.approx(0.2)
    assert isinstance(helper["predicted_ontology_tokens"], list)


def test_meta_transformer_required_mode_rejects_unready_package(tmp_path: Path) -> None:
    runtime_package = _train_runtime_package(tmp_path)
    transformer = MetaTransformer(
        d_shared=24,
        helper_package_path=runtime_package,
        helper_mode="required",
    )

    with pytest.raises(ValueError, match="benchmark-gated package"):
        transformer.forward(
            dino_features=np.zeros(256, dtype=np.float32),
            vla_features=np.zeros(128, dtype=np.float32),
        )
