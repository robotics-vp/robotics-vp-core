"""Tests for workcell environment configuration."""
from __future__ import annotations

import json
import pytest

from src.envs.workcell_env.config import WorkcellEnvConfig, PRESETS


class TestWorkcellEnvConfig:
    """Tests for WorkcellEnvConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WorkcellEnvConfig()
        assert config.topology_type == "ASSEMBLY_BENCH"
        assert config.num_stations == 2
        assert config.num_fixtures == 2
        assert config.num_bins == 4
        assert config.conveyor_enabled is False
        assert config.num_parts == 12
        assert config.max_steps == 200
        assert config.physics_mode == "SIMPLE"
        assert config.tolerance_mm == 2.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WorkcellEnvConfig(
            topology_type="CONVEYOR_LINE",
            num_stations=4,
            conveyor_enabled=True,
            num_parts=20,
        )
        assert config.topology_type == "CONVEYOR_LINE"
        assert config.num_stations == 4
        assert config.conveyor_enabled is True
        assert config.num_parts == 20

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        config = WorkcellEnvConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["topology_type"] == "ASSEMBLY_BENCH"
        assert d["num_stations"] == 2
        assert isinstance(d["part_types"], list)

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "topology_type": "INSPECTION_STATION",
            "num_parts": 8,
            "tolerance_mm": 0.5,
        }
        config = WorkcellEnvConfig.from_dict(d)

        assert config.topology_type == "INSPECTION_STATION"
        assert config.num_parts == 8
        assert config.tolerance_mm == 0.5
        # Defaults should be preserved
        assert config.num_stations == 2

    def test_dict_roundtrip(self) -> None:
        """Test dict serialization roundtrip."""
        config = WorkcellEnvConfig(
            topology_type="TOOL_CABINET",
            num_parts=15,
            part_types=("screw", "bolt", "nut"),
        )
        d = config.to_dict()
        restored = WorkcellEnvConfig.from_dict(d)

        assert config == restored

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        config = WorkcellEnvConfig(
            topology_type="MIXED_WORKCELL",
            conveyor_enabled=True,
        )
        json_str = config.to_json()
        restored = WorkcellEnvConfig.from_json(json_str)

        assert config == restored
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["topology_type"] == "MIXED_WORKCELL"

    def test_presets_exist(self) -> None:
        """Test that presets are defined."""
        assert len(PRESETS) >= 3
        assert "assembly_bench_simple" in PRESETS
        assert "conveyor_sorting" in PRESETS
        assert "inspection_simple" in PRESETS

    def test_preset_values(self) -> None:
        """Test preset configuration values."""
        assembly = PRESETS["assembly_bench_simple"]
        assert assembly.topology_type == "ASSEMBLY_BENCH"
        assert assembly.conveyor_enabled is False

        conveyor = PRESETS["conveyor_sorting"]
        assert conveyor.topology_type == "CONVEYOR_LINE"
        assert conveyor.conveyor_enabled is True

    def test_frozen_config(self) -> None:
        """Test that config is frozen (immutable)."""
        config = WorkcellEnvConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.num_parts = 100


class TestWorkcellEnvConfigLoader:
    """Tests for config loader function."""

    def test_load_from_dict(self) -> None:
        """Test loading config from dict."""
        from src.config.workcell_env_config import load_workcell_env_config

        config = load_workcell_env_config({"num_parts": 25})
        assert config.num_parts == 25

    def test_load_from_preset(self) -> None:
        """Test loading config from preset name."""
        from src.config.workcell_env_config import load_workcell_env_config

        config = load_workcell_env_config(preset="conveyor_sorting")
        assert config.conveyor_enabled is True

    def test_load_with_overrides(self) -> None:
        """Test loading preset with overrides."""
        from src.config.workcell_env_config import load_workcell_env_config

        config = load_workcell_env_config(
            data={"num_parts": 50},
            preset="assembly_bench_simple",
        )
        # Preset value
        assert config.topology_type == "ASSEMBLY_BENCH"
        # Override value
        assert config.num_parts == 50
