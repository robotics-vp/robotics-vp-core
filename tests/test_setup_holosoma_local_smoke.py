from __future__ import annotations

import json
from pathlib import Path

from scripts import setup_holosoma_local_smoke


def _make_holosoma_root(tmp_path: Path) -> Path:
    root = tmp_path / "holosoma"
    for subpath in setup_holosoma_local_smoke.HOLOSOMA_SUBPATHS:
        package_root = root / subpath
        package_root.mkdir(parents=True)
        package_name = package_root.name
        module_dir = package_root / package_name
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_setup_holosoma_local_smoke_writes_path_shim(tmp_path: Path, capsys) -> None:
    holosoma_root = _make_holosoma_root(tmp_path)
    site_packages = tmp_path / "site-packages"

    rc = setup_holosoma_local_smoke.main(
        [
            "--holosoma-root",
            str(holosoma_root),
            "--site-packages-dir",
            str(site_packages),
        ]
    )

    assert rc == 0
    pth_path = site_packages / setup_holosoma_local_smoke.DEFAULT_PTH_NAME
    assert pth_path.exists()
    assert pth_path.read_text(encoding="utf-8").splitlines() == [
        str((holosoma_root / subpath).resolve())
        for subpath in setup_holosoma_local_smoke.HOLOSOMA_SUBPATHS
    ]
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "installed"
    assert report["written"] is True


def test_setup_holosoma_local_smoke_blocks_missing_paths(tmp_path: Path, capsys) -> None:
    site_packages = tmp_path / "site-packages"

    rc = setup_holosoma_local_smoke.main(
        [
            "--holosoma-root",
            str(tmp_path / "missing-holosoma"),
            "--site-packages-dir",
            str(site_packages),
        ]
    )

    assert rc == 2
    assert not (site_packages / setup_holosoma_local_smoke.DEFAULT_PTH_NAME).exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked_missing_paths"
    assert len(report["missing_path_entries"]) == len(setup_holosoma_local_smoke.HOLOSOMA_SUBPATHS)


def test_setup_holosoma_local_smoke_removes_path_shim(tmp_path: Path, capsys) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    pth_path = site_packages / setup_holosoma_local_smoke.DEFAULT_PTH_NAME
    pth_path.write_text("/tmp/holosoma/src/holosoma\n", encoding="utf-8")

    rc = setup_holosoma_local_smoke.main(
        [
            "--site-packages-dir",
            str(site_packages),
            "--remove",
        ]
    )

    assert rc == 0
    assert not pth_path.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "removed"
    assert report["removed"] is True
