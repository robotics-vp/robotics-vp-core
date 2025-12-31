import json
import tempfile
from pathlib import Path

from scripts.export_lsd_vector_scene_dataset import ExportConfig, export_dataset


def test_export_lsd_with_motion_hierarchy():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExportConfig(
            num_scenes=1,
            episodes_per_scene=1,
            max_steps=6,
            output_path=tmpdir,
            shard_size=10,
            seed=7,
            enable_motion_hierarchy=True,
            num_humans_range=(1, 1),
            num_forklifts_range=(0, 0),
        )

        export_dataset(config, verbose=False)

        index_path = Path(tmpdir) / "index.json"
        with index_path.open("r") as f:
            index = json.load(f)

        shard_path = Path(tmpdir) / index["shards"][0]["file_path"]
        with shard_path.open("r") as f:
            episodes = json.load(f)

        assert episodes
        episode = episodes[0]
        assert "trajectory" in episode
        assert "agent_labels" in episode
        assert "agent_trajectories" in episode
        assert "motion_hierarchy" in episode

        mh = episode["motion_hierarchy"]
        assert "node_labels" in mh
        assert "hierarchy" in mh
        assert len(mh["node_labels"]) == len(mh["hierarchy"])
        assert len(mh["hierarchy"]) == len(mh["hierarchy"][0])
