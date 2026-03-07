from src.objectives.profile_loader import load_contract_profile
from src.objectives.runtime_builder import (
    ObjectiveRuntimeBuilder,
    ObjectiveRuntimeRecord,
    ObjectiveRuntimeWindow,
    SourceDomain,
)


def test_objective_runtime_builder_emits_context_and_provenance():
    builder = ObjectiveRuntimeBuilder()
    record = ObjectiveRuntimeRecord(
        task_id="shadow_kitting",
        episode_id="ep_shadow_001",
        env_id="workcell_simple",
        world_id="workcell_assembly_bench_simple",
        robot_id="shadow_sim_arm_v1",
        source_domain=SourceDomain.SYNTHETIC,
        seed=7,
        run_id="shadow_run",
        timestamp="2026-01-01T00:00:00+00:00",
        episode_metrics={
            "items_completed": 3,
            "duration_s": 720.0,
            "error_rate": 0.05,
            "energy_wh": 1.5,
            "energy_wh_per_unit": 0.5,
            "safety_score": 0.92,
        },
        reward_components={
            "mpl_component": 0.7,
            "delta_errors": 0.05,
            "energy_penalty": 0.2,
            "safety_bonus": 0.1,
        },
        telemetry={"trust_score": 0.88, "uncertainty": 0.12},
        windows=(
            ObjectiveRuntimeWindow(
                window_id="window_000_001",
                start_step=0,
                end_step=1,
                metrics={
                    "items_completed": 2,
                    "duration_s": 480.0,
                    "error_rate": 0.0,
                    "energy_wh": 0.8,
                    "energy_wh_per_unit": 0.4,
                    "safety_score": 0.95,
                },
                telemetry={"trust_score": 0.91, "uncertainty": 0.08},
            ),
        ),
    )

    tensor = builder.build(record)
    assert tensor.context["task_id"] == "shadow_kitting"
    assert tensor.context["source_domain"] == SourceDomain.SYNTHETIC.value
    assert tensor.context["schema_version_hash"]
    assert tensor.provenance["runtime_record_hash"]
    assert tensor.values.shape[-1] == 4

    window_payloads = builder.build_window_tensors(record)
    assert len(window_payloads) == 1
    assert window_payloads[0]["window"]["window_id"] == "window_000_001"


def test_contract_profile_loader_and_compile_boundary():
    builder = ObjectiveRuntimeBuilder()
    profile = load_contract_profile("balanced_contract")
    tensor = builder.build(
        ObjectiveRuntimeRecord(
            task_id="shadow_kitting",
            episode_id="ep_shadow_002",
            env_id="workcell_simple",
            world_id="workcell_assembly_bench_simple",
            robot_id="shadow_sim_arm_v1",
            source_domain=SourceDomain.SYNTHETIC,
            seed=11,
            run_id="shadow_run",
            timestamp="2026-01-01T00:10:00+00:00",
            episode_metrics={
                "items_completed": 3,
                "duration_s": 720.0,
                "error_rate": 0.10,
                "energy_wh": 2.1,
                "energy_wh_per_unit": 0.7,
                "safety_score": 0.86,
            },
            telemetry={"trust_score": 0.82, "uncertainty": 0.20},
        )
    )
    compile_result = builder.compile_contract(tensor, profile.profile)

    assert profile.profile.profile_id == "balanced_contract"
    assert profile.soft_constraints
    assert compile_result.objective_profile_id == "balanced_contract"
    assert compile_result.scalarization_boundary == "contract_boundary"
    assert isinstance(compile_result.scalar_reward, float)
