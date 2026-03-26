from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.ontology.datapack_registry import register_datapack_configs
from src.ontology.models import Datapack
from src.ontology.store import OntologyStore


def test_register_datapack_configs_upserts_existing_truth(tmp_path) -> None:
    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.append_datapacks(
        [
            Datapack(
                datapack_id="dp_existing",
                source_type="holosoma",
                task_id="task_a",
                modality="motion",
                storage_uri="data/old_clip.npz",
                quality_score=0.1,
                novelty_score=0.0,
                metadata={
                    "description": "stale",
                    "scene_tracks_backend": "passthrough",
                    "execution_preconditions": {"ready": False},
                },
            )
        ]
    )

    register_datapack_configs(
        store,
        task_id="task_a",
        datapack_configs=[
            DatapackConfig(
                id="dp_existing",
                description="fresh",
                motion_clips=[MotionClipSpec(path="data/new_clip.npz", weight=1.0)],
                quality_score=0.75,
                novelty_score=0.3,
                tags=["warehouse"],
                metadata={
                    "scene_tracks_backend": "real",
                    "execution_preconditions": {"ready": True},
                    "future_training_signals": {"semantic_grounding_non_heuristic": True},
                },
            )
        ],
    )

    [datapack] = store.list_datapacks(task_id="task_a")
    assert datapack.storage_uri == "data/new_clip.npz"
    assert datapack.quality_score == 0.75
    assert datapack.novelty_score == 0.3
    assert datapack.metadata["description"] == "fresh"
    assert datapack.metadata["scene_tracks_backend"] == "real"
    assert datapack.metadata["execution_preconditions"]["ready"] is True
    assert datapack.tags["semantic_tags"] == ["warehouse"]
