import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_train_sac_help_includes_objective_scalarizer():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "train_sac.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--objective-scalarizer" in proc.stdout


def test_train_ppo_help_includes_objective_scalarizer():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "train_ppo.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--objective-scalarizer" in proc.stdout
