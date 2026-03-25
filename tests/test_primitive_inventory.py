"""Tests for environment primitive inventory (B3)."""
import pytest

from src.envs.primitive_inventory import (
    EnvPrimitive,
    EnvPrimitiveInventory,
    for_env,
    get_inventory,
    list_registered_env_ids,
    register_inventory,
)


def test_builtin_inventories_registered():
    registered = list_registered_env_ids()
    assert "drawer_vase" in registered
    assert "dishwashing" in registered
    assert "workcell" in registered


def test_drawer_vase_primitives():
    inv = for_env("drawer_vase")
    assert inv.env_family == "drawer_vase"
    assert len(inv.primitives) == 8
    assert inv.has_primitive("locate_handle")
    assert inv.has_primitive("collision_avoidance")
    assert not inv.has_primitive("nonexistent")


def test_primitives_by_category():
    inv = for_env("drawer_vase")
    manip = inv.primitives_by_category("manipulation")
    assert len(manip) >= 3
    assert all(p.category == "manipulation" for p in manip)


def test_serialisation_round_trip():
    inv = for_env("dishwashing")
    d = inv.to_dict()
    inv2 = EnvPrimitiveInventory.from_dict(d)
    assert inv2.env_id == "dishwashing"
    assert len(inv2.primitives) == len(inv.primitives)
    assert inv2.object_families == inv.object_families


def test_custom_registration():
    custom = EnvPrimitiveInventory(
        env_id="test_custom",
        env_family="test",
        primitives=[EnvPrimitive("foo", "manipulation", "Foo")],
    )
    register_inventory(custom)
    assert get_inventory("test_custom") is not None
    assert for_env("test_custom").primitives[0].primitive_id == "foo"


def test_for_env_raises_on_missing():
    with pytest.raises(KeyError, match="No primitive inventory"):
        for_env("nonexistent_env_xyz")
