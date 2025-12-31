from __future__ import annotations

import dataclasses
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import DatapackBundle, DatapackConfig, DatapackProvider, MotionClipSpec
from src.motor_backend.holosoma_reward_terms import analyze_anti_reward_hacking
from src.objectives.economic_objective import (
    CompiledRewardOverlay,
    EconomicObjectiveSpec,
    compile_economic_overlay,
)

try:  # pragma: no cover
    import holosoma
    import holosoma_inference
    import holosoma_retargeting
except ImportError:  # pragma: no cover
    holosoma = None
    holosoma_inference = None
    holosoma_retargeting = None


def ensure_holosoma_available() -> None:
    if holosoma is None:
        raise RuntimeError(
            "Holosoma backend requested but 'holosoma' is not installed. "
            "Install with `pip install -r requirements-holosoma.txt` or the holosoma extra."
        )


@dataclass(frozen=True)
class HolosomaTaskSpec:
    exp_name: str
    simulator: str
    task_name: str
    description: str = ""


# Holosoma experiment presets live in `holosoma.config_values.experiment.DEFAULTS` and
# reference per-task config fragments under `holosoma/config_values/{loco,wbt}`.
# TODO: confirm the exact preset names you want to target for each task_id.
HOLOSOMA_TASK_MAP: dict[str, HolosomaTaskSpec] = {
    "humanoid_locomotion_g1": HolosomaTaskSpec(
        exp_name="g1_29dof_fast_sac",
        simulator="isaacgym",
        task_name="locomotion",
        description="G1 locomotion (FastSAC) using holosoma.config_values.loco.g1 presets.",
    ),
    "humanoid_locomotion_t1": HolosomaTaskSpec(
        exp_name="t1_29dof_fast_sac",
        simulator="isaacgym",
        task_name="locomotion",
        description="T1 locomotion (FastSAC) using holosoma.config_values.loco.t1 presets.",
    ),
    "humanoid_wbt_g1": HolosomaTaskSpec(
        exp_name="g1_29dof_wbt_fast_sac",
        simulator="isaacsim",
        task_name="wbt",
        description="G1 whole-body tracking (FastSAC) using holosoma.config_values.wbt.g1 presets.",
    ),
}


@dataclass(frozen=True)
class HolosomaRunResult:
    policy_id: str
    raw_metrics: Mapping[str, float]


class HolosomaRunner(Protocol):
    def train(
        self,
        task_spec: HolosomaTaskSpec,
        overlay: CompiledRewardOverlay,
        datapack_bundle: DatapackBundle,
        num_envs: int,
        max_steps: int,
        seed: int | None,
    ) -> HolosomaRunResult:
        ...

    def evaluate(
        self,
        policy_id: str,
        task_spec: HolosomaTaskSpec,
        overlay: CompiledRewardOverlay,
        num_episodes: int,
        seed: int | None,
    ) -> HolosomaRunResult:
        ...

    def load_policy(self, policy_id: str) -> Any:
        ...


class DefaultHolosomaRunner:
    def __init__(self, base_log_dir: Path | None = None) -> None:
        self._base_log_dir = base_log_dir

    def train(
        self,
        task_spec: HolosomaTaskSpec,
        overlay: CompiledRewardOverlay,
        datapack_bundle: DatapackBundle,
        num_envs: int,
        max_steps: int,
        seed: int | None,
    ) -> HolosomaRunResult:
        ensure_holosoma_available()
        config = self._build_experiment_config(task_spec, num_envs, max_steps, seed)
        config = self._apply_reward_overlay(config, overlay)
        config = self._apply_datapack_overrides(config, datapack_bundle)
        config = self._apply_motion_overrides(config, datapack_bundle)

        project_dir = self._project_dir(config)
        project_dir.mkdir(parents=True, exist_ok=True)
        before = {p for p in project_dir.iterdir() if p.is_dir()}

        self._ensure_pythonpath()
        from holosoma.holosoma.train_agent import train as holosoma_train

        holosoma_train(config)

        after = {p for p in project_dir.iterdir() if p.is_dir()}
        run_dir = self._resolve_new_dir(before, after) or project_dir
        policy_id = self._find_latest_checkpoint(run_dir)

        raw_metrics = {
            "train_steps": float(max_steps),
            "num_envs": float(num_envs),
        }
        raw_metrics.update(self._collect_log_metrics(run_dir, config))
        if seed is not None:
            raw_metrics["seed"] = float(seed)
        return HolosomaRunResult(policy_id=policy_id, raw_metrics=raw_metrics)

    def evaluate(
        self,
        policy_id: str,
        task_spec: HolosomaTaskSpec,
        overlay: CompiledRewardOverlay,
        num_episodes: int,
        seed: int | None,
    ) -> HolosomaRunResult:
        ensure_holosoma_available()
        if num_episodes <= 0:
            return HolosomaRunResult(policy_id=policy_id, raw_metrics={})
        try:
            from holosoma.holosoma.eval_agent import run_eval_with_tyro
            from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
        except Exception as exc:
            raise RuntimeError("Holosoma evaluation entrypoints are unavailable.") from exc

        checkpoint_cfg = CheckpointConfig(checkpoint=policy_id)
        saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
        eval_cfg = saved_cfg.get_eval_config()
        if seed is not None:
            eval_cfg = dataclasses.replace(eval_cfg, training=dataclasses.replace(eval_cfg.training, seed=seed))
        eval_cfg = self._apply_reward_overlay(eval_cfg, overlay)

        base_dir = Path(getattr(eval_cfg.logger, "base_dir", "logs"))
        project = eval_cfg.training.project or getattr(eval_cfg.logger, "project", None) or "default_project"
        eval_project_dir = base_dir / project
        eval_project_dir.mkdir(parents=True, exist_ok=True)
        before = {p for p in eval_project_dir.iterdir() if p.is_dir()}

        # TODO: map num_episodes into evaluation stopping criteria for Holosoma.
        run_eval_with_tyro(eval_cfg, checkpoint_cfg, saved_cfg, saved_wandb_path)

        after = {p for p in eval_project_dir.iterdir() if p.is_dir()}
        run_dir = self._resolve_new_dir(before, after) or eval_project_dir

        raw_metrics = {
            "num_episodes": float(num_episodes),
        }
        raw_metrics.update(self._collect_log_metrics(run_dir, eval_cfg))
        if seed is not None:
            raw_metrics["seed"] = float(seed)
        return HolosomaRunResult(policy_id=policy_id, raw_metrics=raw_metrics)

    def load_policy(self, policy_id: str) -> Any:
        return HolosomaPolicyHandle(policy_id)

    def _build_experiment_config(self, task_spec: HolosomaTaskSpec, num_envs: int, max_steps: int, seed: int | None):
        from holosoma.config_values import experiment as holo_experiment
        from holosoma.config_values import logger as holo_logger
        from holosoma.config_values import simulator as holo_simulator

        if task_spec.exp_name not in holo_experiment.DEFAULTS:
            raise ValueError(f"Unknown Holosoma experiment preset: {task_spec.exp_name}")
        if task_spec.simulator not in holo_simulator.DEFAULTS:
            raise ValueError(f"Unknown Holosoma simulator preset: {task_spec.simulator}")

        base_cfg = holo_experiment.DEFAULTS[task_spec.exp_name]
        sim_cfg = holo_simulator.DEFAULTS[task_spec.simulator]
        logger_cfg = holo_logger.disabled
        if self._base_log_dir is not None:
            logger_cfg = dataclasses.replace(logger_cfg, base_dir=str(self._base_log_dir))

        training_cfg = dataclasses.replace(
            base_cfg.training,
            num_envs=num_envs,
            seed=seed if seed is not None else base_cfg.training.seed,
            export_onnx=True,
        )
        algo_cfg = base_cfg.algo
        algo_inner = algo_cfg.config
        if hasattr(algo_inner, "num_learning_iterations"):
            algo_inner = dataclasses.replace(algo_inner, num_learning_iterations=max_steps)
            algo_cfg = dataclasses.replace(algo_cfg, config=algo_inner)

        return dataclasses.replace(
            base_cfg,
            simulator=sim_cfg,
            logger=logger_cfg,
            training=training_cfg,
            algo=algo_cfg,
        )

    def _apply_reward_overlay(self, config: Any, overlay: CompiledRewardOverlay):
        if not overlay.reward_scales:
            return config
        from holosoma.config_types.reward import RewardTermCfg

        reward_cfg = getattr(config, "reward", None)
        if reward_cfg is None:
            return config
        terms = dict(getattr(reward_cfg, "terms", {}) or {})
        terms["economic_overlay"] = RewardTermCfg(
            func="src.motor_backend.holosoma_reward_terms:economic_overlay_reward",
            params={"reward_scales": dict(overlay.reward_scales)},
            weight=1.0,
            tags=["econ"],
        )
        reward_cfg = dataclasses.replace(reward_cfg, terms=terms)
        return dataclasses.replace(config, reward=reward_cfg)

    def _apply_datapack_overrides(self, config: Any, datapack_bundle: DatapackBundle):
        randomization = datapack_bundle.randomization_overrides
        curriculum = datapack_bundle.curriculum_overrides

        if randomization and hasattr(config, "randomization"):
            updated_randomization = _apply_dataclass_overrides(config.randomization, randomization)
            config = dataclasses.replace(config, randomization=updated_randomization)
        if curriculum and hasattr(config, "curriculum"):
            updated_curriculum = _apply_dataclass_overrides(config.curriculum, curriculum)
            config = dataclasses.replace(config, curriculum=updated_curriculum)
        return config

    def _apply_motion_overrides(self, config: Any, datapack_bundle: DatapackBundle):
        if not datapack_bundle.motion_clips:
            return config
        command_cfg = getattr(config, "command", None)
        if command_cfg is None:
            return config
        setup_terms = getattr(command_cfg, "setup_terms", {}) or {}
        motion_term = setup_terms.get("motion_command")
        if motion_term is None:
            return config
        params = dict(getattr(motion_term, "params", {}) or {})
        motion_cfg = params.get("motion_config")
        if motion_cfg is None:
            return config

        primary_motion = _select_primary_motion(datapack_bundle.motion_clips)
        motion_cfg = dataclasses.replace(motion_cfg, motion_file=primary_motion.path)
        params["motion_config"] = motion_cfg
        motion_term = dataclasses.replace(motion_term, params=params)
        setup_terms = dict(setup_terms)
        setup_terms["motion_command"] = motion_term
        command_cfg = dataclasses.replace(command_cfg, setup_terms=setup_terms)
        return dataclasses.replace(config, command=command_cfg)

    def _collect_log_metrics(self, run_dir: Path, config: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        log_file = self._find_latest_log(run_dir)
        if log_file is not None:
            metrics.update(_parse_log_metrics(log_file))
        step_dt = _estimate_step_dt(config)
        if step_dt and metrics.get("mean_episode_length"):
            metrics["mean_episode_length_s"] = metrics["mean_episode_length"] * step_dt
        if metrics.get("mean_episode_length_s"):
            success_rate = metrics.get("success_rate", 1.0)
            metrics.setdefault(
                "mpl_units_per_hour",
                (success_rate * 3600.0) / max(metrics["mean_episode_length_s"], 1e-6),
            )
        if metrics.get("success_rate") is not None:
            metrics.setdefault("error_rate", max(0.0, 1.0 - metrics["success_rate"]))
        anti_report = analyze_anti_reward_hacking(metrics)
        if anti_report.summary_metrics:
            for key, value in anti_report.summary_metrics.items():
                metrics.setdefault(key, value)
        if anti_report.is_suspicious:
            metrics["anti_reward_hacking_suspicious"] = 1.0
            if anti_report.reasons:
                metrics["anti_reward_hacking_reason"] = "; ".join(anti_report.reasons)
        return metrics

    def _find_latest_log(self, run_dir: Path) -> Path | None:
        candidates = [p for p in run_dir.glob("*.log") if p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _project_dir(self, config: Any) -> Path:
        base_dir = Path(getattr(config.logger, "base_dir", "logs"))
        project = config.training.project or getattr(config.logger, "project", None) or "default_project"
        return base_dir / project

    def _resolve_new_dir(self, before: set[Path], after: set[Path]) -> Path | None:
        new_dirs = list(after - before)
        if new_dirs:
            return max(new_dirs, key=lambda p: p.stat().st_mtime)
        if after:
            return max(after, key=lambda p: p.stat().st_mtime)
        return None

    def _find_latest_checkpoint(self, run_dir: Path) -> str:
        pattern = re.compile(r"model_(\d+)\.(pt|onnx)$")
        candidates = []
        for path in run_dir.glob("model_*.pt"):
            match = pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        for path in run_dir.glob("model_*.onnx"):
            match = pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            _, best = max(candidates, key=lambda x: x[0])
            return str(best)
        return str(run_dir)

    def _ensure_pythonpath(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


class HolosomaPolicyHandle:
    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        if not policy_id.endswith(".onnx"):
            return
        if holosoma_inference is None:
            return
        try:
            import onnxruntime as ort
        except Exception:
            return
        self._session = ort.InferenceSession(policy_id)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    def act(self, obs: Any) -> Any:
        if self._session is None:
            raise RuntimeError("Holosoma policy handle is not initialized for inference.")
        if isinstance(obs, dict):
            input_feed = {name: obs[name] for name in self._input_names if name in obs}
        else:
            if not self._input_names:
                raise RuntimeError("Holosoma policy handle has no inputs.")
            input_feed = {self._input_names[0]: obs}
        outputs = self._session.run(self._output_names, input_feed)
        return outputs[0] if len(outputs) == 1 else outputs

    def step(self, obs: Any) -> Any:
        return self.act(obs)


class HolosomaBackend:
    def __init__(
        self,
        econ_meter: EconomicMeter,
        datapack_provider: DatapackProvider,
        runner: HolosomaRunner | None = None,
    ) -> None:
        self._econ_meter = econ_meter
        self._datapack_provider = datapack_provider
        self._runner = runner or DefaultHolosomaRunner()

    def train_policy(
        self,
        task_id: str,
        objective: EconomicObjectiveSpec,
        datapack_ids: Sequence[str],
        num_envs: int,
        max_steps: int,
        datapack_configs: Sequence[DatapackConfig] | None = None,
        seed: int | None = None,
    ) -> MotorTrainingResult:
        ensure_holosoma_available()
        task_spec = self._resolve_task(task_id)
        overlay = compile_economic_overlay(objective)
        datapack_bundle = self._datapack_provider.resolve(task_id, datapack_ids, datapack_configs)
        result = self._runner.train(task_spec, overlay, datapack_bundle, num_envs, max_steps, seed)
        econ_metrics = self._econ_meter.summarize(result.raw_metrics)
        econ_metrics = _apply_anti_reward_hacking_flags(econ_metrics, result.raw_metrics)
        return MotorTrainingResult(
            policy_id=result.policy_id,
            raw_metrics=result.raw_metrics,
            econ_metrics=econ_metrics,
        )

    def evaluate_policy(
        self,
        policy_id: str,
        task_id: str,
        objective: EconomicObjectiveSpec,
        num_episodes: int,
        seed: int | None = None,
    ) -> MotorEvalResult:
        ensure_holosoma_available()
        task_spec = self._resolve_task(task_id)
        overlay = compile_economic_overlay(objective)
        result = self._runner.evaluate(policy_id, task_spec, overlay, num_episodes, seed)
        econ_metrics = self._econ_meter.summarize(result.raw_metrics)
        econ_metrics = _apply_anti_reward_hacking_flags(econ_metrics, result.raw_metrics)
        return MotorEvalResult(
            policy_id=result.policy_id,
            raw_metrics=result.raw_metrics,
            econ_metrics=econ_metrics,
        )

    def deploy_policy_handle(self, policy_id: str) -> Any:
        ensure_holosoma_available()
        return self._runner.load_policy(policy_id)

    def _resolve_task(self, task_id: str) -> HolosomaTaskSpec:
        if task_id not in HOLOSOMA_TASK_MAP:
            raise ValueError(
                "Unknown Holosoma task_id: "
                f"{task_id}. Known: {sorted(HOLOSOMA_TASK_MAP)}. "
                "Update HOLOSOMA_TASK_MAP to add a mapping to a Holosoma experiment preset."
            )
        return HOLOSOMA_TASK_MAP[task_id]


def _select_primary_motion(clips: Sequence[MotionClipSpec]) -> MotionClipSpec:
    if not clips:
        raise ValueError("No motion clips available for Holosoma override.")
    return max(clips, key=lambda clip: clip.weight)


def _apply_dataclass_overrides(obj: Any, overrides: Mapping[str, Any]) -> Any:
    if not dataclasses.is_dataclass(obj) or not overrides:
        return obj
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            updates[key] = _apply_dataclass_overrides(current, value)
        else:
            updates[key] = value
    if not updates:
        return obj
    return dataclasses.replace(obj, **updates)


def _parse_log_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    key_val = re.compile(r"([A-Za-z0-9_./\- ]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    mean_reward_re = re.compile(r"Mean reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    mean_length_re = re.compile(r"Mean episode length:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = mean_reward_re.search(line)
        if match:
            metrics["mean_reward"] = float(match.group(1))
        match = mean_length_re.search(line)
        if match:
            metrics["mean_episode_length"] = float(match.group(1))
        match = key_val.search(line)
        if match:
            key = match.group(1).strip()
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            metrics[key] = value

    success_key = next((k for k in metrics if "success_rate" in k.lower()), None)
    if success_key:
        metrics.setdefault("success_rate", metrics[success_key])

    return metrics


def _estimate_step_dt(config: Any) -> float:
    try:
        sim_cfg = config.simulator.config.sim
        fps = float(sim_cfg.fps)
        control_decimation = float(sim_cfg.control_decimation)
        if fps > 0:
            return control_decimation / fps
    except Exception:
        return 0.0
    return 0.0


def _apply_anti_reward_hacking_flags(
    econ_metrics: Mapping[str, float],
    raw_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not raw_metrics:
        return econ_metrics
    suspicious = raw_metrics.get("anti_reward_hacking_suspicious")
    if suspicious is None:
        return econ_metrics
    merged: dict[str, Any] = dict(econ_metrics)
    try:
        merged["anti_reward_hacking_suspicious"] = float(suspicious)
    except (TypeError, ValueError):
        merged["anti_reward_hacking_suspicious"] = 1.0 if suspicious else 0.0
    if "anti_reward_hacking_reason" in raw_metrics:
        merged["anti_reward_hacking_reason"] = raw_metrics.get("anti_reward_hacking_reason")
    return merged
