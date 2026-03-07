"""Objective specifications, tensors, and scalarization helpers."""

from src.objectives.compiler import ObjectiveCompiler  # noqa: F401
from src.objectives.economic_objective import (  # noqa: F401
    CompiledRewardOverlay,
    EconomicObjectiveSpec,
    compile_economic_overlay,
)
from src.objectives.frontier import ParetoFrontierTracker  # noqa: F401
from src.objectives.loader import load_objective_spec  # noqa: F401
from src.objectives.profile import ObjectiveProfile  # noqa: F401
from src.objectives.profile_loader import (  # noqa: F401
    ObjectiveContractProfile,
    load_all_contract_profiles,
    load_contract_profile,
)
from src.objectives.runtime_builder import (  # noqa: F401
    ObjectiveCompileResult,
    ObjectiveRuntimeBuilder,
    ObjectiveRuntimeRecord,
    ObjectiveRuntimeWindow,
    SourceDomain,
    build_runtime_schema,
    summarize_objective_tensor,
)
from src.objectives.schema import ObjectiveTensorSchema  # noqa: F401
from src.objectives.serialization import (  # noqa: F401
    load_objective_tensor_npz,
    objective_tensor_from_npz_dict,
    objective_tensor_to_npz_dict,
    save_objective_tensor_npz,
)
from src.objectives.tensor import ObjectiveTensor, objective_tensor_from_axes  # noqa: F401
