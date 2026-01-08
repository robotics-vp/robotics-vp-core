from src.policies.sampler_weights import HeuristicSamplerWeightPolicy
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.datapack_engine import DatapackEngine, DatapackSignals
from src.orchestrator.economic_controller import EconomicController, EconSignals
from src.orchestrator.task_graph import TaskGraph, TaskNode
from src.orchestrator.ontology import EnvironmentOntology
from src.config.econ_params import EconParams
from src.valuation.datapack_repo import DataPackRepo


def test_sampler_weights_epiplexity_roi():
    policy = HeuristicSamplerWeightPolicy()
    episodes = [
        {"descriptor": {"pack_id": "ep0", "w_epi": 0.1}, "recap_weight_multiplier": 1.0},
        {"descriptor": {"pack_id": "ep1", "w_epi": 0.5}, "recap_weight_multiplier": 1.0},
    ]
    features = policy.build_features(episodes)
    weights = policy.evaluate(features, strategy="epiplexity_roi")
    assert weights["ep1"] > weights["ep0"]


def test_semantic_orchestrator_epiplexity_term():
    econ_params = EconParams(
        price_per_unit=1.0,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.1,
        time_step_s=1.0,
        base_rate=1.0,
        p_min=0.01,
        k_err=0.1,
        q_speed=1.0,
        q_care=1.0,
        care_cost=0.1,
        max_steps=10,
        max_catastrophic_errors=1,
        max_error_rate_sla=0.5,
        min_steps_for_sla=1,
        zero_throughput_patience=1,
        preset="toy",
    )
    econ = EconomicController(econ_params)
    repo = DataPackRepo(base_dir="data/datapacks")
    datapack_engine = DatapackEngine(repo)
    task_graph = TaskGraph(root=TaskNode(task_id="root", name="root"))
    ontology = EnvironmentOntology(ontology_id="test", name="test")

    orchestrator = SemanticOrchestrator(
        econ_controller=econ,
        datapack_engine=datapack_engine,
        task_graph=task_graph,
        ontology=ontology,
        config={"use_epiplexity_term": True, "epi_alpha": 0.5},
    )

    econ_signals = EconSignals()
    datapack_signals = DatapackSignals(mean_delta_epi_per_flop=0.2)
    plan = orchestrator.build_update_plan(econ_signals, datapack_signals)

    assert "epiplexity_term" in plan.cross_module_constraints
    assert plan.cross_module_constraints["epiplexity_term"]["epi_alpha"] == 0.5
