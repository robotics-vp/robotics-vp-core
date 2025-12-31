"""Interface tests for Holosoma motor backend."""
import pytest

pytest.importorskip("holosoma")

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.datapacks import DatapackConfig, DatapackProvider, MotionClipSpec
from src.motor_backend.holosoma_backend import HolosomaBackend, HolosomaRunResult
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.ontology.models import Robot, Task
from src.ontology.store import OntologyStore


class DummyPolicyHandle:
    def act(self, obs):
        return obs

    def step(self, obs):
        return obs


class DummyRunner:
    def __init__(self):
        self.train_calls = []
        self.eval_calls = []

    def train(self, task_spec, overlay, datapack_bundle, num_envs, max_steps, seed):
        self.train_calls.append((task_spec, overlay, datapack_bundle, num_envs, max_steps, seed))
        return HolosomaRunResult(
            policy_id="policy.onnx",
            raw_metrics={"mpl_units_per_hour": 80.0, "wage_parity": 1.2},
        )

    def evaluate(self, policy_id, task_spec, overlay, num_episodes, seed):
        self.eval_calls.append((policy_id, task_spec, overlay, num_episodes, seed))
        return HolosomaRunResult(
            policy_id=policy_id,
            raw_metrics={"mpl_units_per_hour": 90.0, "wage_parity": 1.3},
        )

    def load_policy(self, policy_id):
        return DummyPolicyHandle()


def test_holosoma_backend_interface(monkeypatch, tmp_path):
    import src.motor_backend.holosoma_backend as holosoma_backend

    monkeypatch.setattr(holosoma_backend, "holosoma", object())

    store = OntologyStore(root_dir=tmp_path / "ontology")
    task = Task(
        task_id="humanoid_locomotion_g1",
        name="Locomotion",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id="robot_test", name="Robot", energy_cost_per_wh=0.12)
    econ_meter = EconomicMeter(task=task, robot=robot)
    runner = DummyRunner()
    backend = HolosomaBackend(
        econ_meter=econ_meter,
        datapack_provider=DatapackProvider(store),
        runner=runner,
    )

    objective = EconomicObjectiveSpec(mpl_weight=2.0, energy_weight=0.5, error_weight=1.0)
    datapack_configs = [
        DatapackConfig(
            id="dp_loco_01",
            description="Test locomotion pack",
            motion_clips=[MotionClipSpec(path="data/mocap/test_clip.npz", weight=1.0)],
            domain_randomization={"friction_range": [0.5, 1.0]},
            curriculum={"initial_difficulty": 0.1},
        )
    ]
    train_result = backend.train_policy(
        task_id="humanoid_locomotion_g1",
        objective=objective,
        datapack_ids=[cfg.id for cfg in datapack_configs],
        datapack_configs=datapack_configs,
        num_envs=4,
        max_steps=10,
        seed=123,
    )
    assert train_result.policy_id == "policy.onnx"
    assert train_result.econ_metrics["mpl_units_per_hour"] == 80.0
    assert runner.train_calls, "Expected Holosoma runner train() to be called"
    _, overlay, bundle, *_ = runner.train_calls[0]
    assert overlay.reward_scales["mpl_per_hour"] == 2.0
    assert overlay.reward_scales["energy_kwh"] == -0.5
    assert overlay.reward_scales["error_rate"] == -1.0
    assert any(clip.path == "data/mocap/test_clip.npz" for clip in bundle.motion_clips)
    assert bundle.randomization_overrides["friction_range"] == [0.5, 1.0]
    assert bundle.curriculum_overrides["initial_difficulty"] == 0.1

    eval_result = backend.evaluate_policy(
        policy_id=train_result.policy_id,
        task_id="humanoid_locomotion_g1",
        objective=objective,
        num_episodes=2,
        seed=456,
    )
    assert eval_result.policy_id == "policy.onnx"
    assert eval_result.econ_metrics["mpl_units_per_hour"] == 90.0

    handle = backend.deploy_policy_handle(train_result.policy_id)
    assert hasattr(handle, "act")
    assert hasattr(handle, "step")
