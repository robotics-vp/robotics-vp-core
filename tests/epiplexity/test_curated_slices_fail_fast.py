import sys

import pytest

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta
from scripts import run_epiplexity_curated_slices


def test_curated_slices_requires_raw_data_path(tmp_path, monkeypatch):
    repo = DataPackRepo(base_dir=str(tmp_path))
    datapack = DataPackMeta(task_name="drawer_vase")
    repo.append(datapack)

    output_dir = tmp_path / "out"
    argv = [
        "run_epiplexity_curated_slices",
        "--datapack-dir",
        str(tmp_path),
        "--task",
        "drawer_vase",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError) as excinfo:
        run_epiplexity_curated_slices.main()

    msg = str(excinfo.value)
    assert "raw_data_path_nonnull=0" in msg
    assert "total=1" in msg
    assert "frames for vision_rgb" in msg
    assert "scene_tracks for geometry_bev" in msg
