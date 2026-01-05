# Third-Party Notices

This file documents third-party code, assets, and design patterns used in this project.

## Workcell Environment Suite

The manufacturing/workcell environment suite draws inspiration and patterns from several open-source robotics projects. No code was directly copied; only design patterns and interface conventions were referenced.

### Safe / Permissive Sources (Referenced for Patterns)

#### robosuite (ARISE Initiative)
- **Repository**: https://github.com/ARISE-Initiative/robosuite
- **License**: Permissive (verify before vendoring)
- **Usage**: Referenced for:
  - Task template patterns (pick-place, assembly primitives)
  - Reward shaping conventions
  - Observation space design patterns
  - Domain randomization approaches
- **No code copied**: Only design patterns referenced

#### robomimic (ARISE Initiative)
- **Repository**: https://github.com/ARISE-Initiative/robomimic
- **License**: MIT
- **Usage**: Referenced for:
  - Dataset schema conventions
  - Trajectory packing patterns
  - Train/eval split logic
- **No code copied**: Only design patterns referenced

#### Meta-World (Farama Foundation)
- **Repository**: https://github.com/Farama-Foundation/Metaworld
- **License**: MIT
- **Usage**: Referenced for:
  - Task list coverage inspiration
  - Success metric patterns
  - Reward template structures
- **No code copied**: Only design patterns referenced

### Conditionally Usable Sources (Plugin-Only)

The following sources have licensing constraints that require them to be treated as optional plugins, not hard dependencies:

#### Isaac Lab
- **Repository**: https://github.com/isaac-sim/IsaacLab
- **License**: BSD-3 (code), but Isaac Sim has additional/proprietary terms
- **Status**: Optional plugin backend only
- **Usage**: May be used as optional physics backend via `physics_mode="ISAAC"`
- **Constraints**: Isaac Sim licensing terms apply; not a hard dependency

#### ManiSkill / ManiSkill2
- **Repository**: https://github.com/haosulab/ManiSkill
- **License**: Code is permissive; assets are CC BY-NC 4.0
- **Status**: Reference only
- **Constraints**: Assets cannot be imported into core repo
- **Usage**: Task interface patterns referenced only

### Excluded Sources (Restrictive Licenses)

The following sources have restrictive licenses. Only papers/ideas were referenced; NO code was copied or vendored:

#### MimicGen / DexMimicGen (NVIDIA)
- **Repositories**:
  - https://github.com/NVlabs/mimicgen
  - https://github.com/NVlabs/dexmimicgen
- **License**: NVIDIA Source Code License (restrictive)
- **Status**: DO NOT vendor code
- **Usage**: Academic papers referenced for conceptual understanding only

#### RLBench (Imperial College)
- **Repository**: https://github.com/stepjam/RLBench
- **License**: Custom Imperial license
- **Status**: DO NOT vendor code
- **Usage**: Not directly used

## License Verification Checklist

Before vendoring any external code:

1. [ ] Record source in this file
2. [ ] Keep vendored code in `third_party/` directory
3. [ ] Include original LICENSE file
4. [ ] Do not import non-commercial assets into core runtime
5. [ ] Treat Isaac integration as optional plugin only
6. [ ] Never copy NVIDIA Source Code License code

## Contact

For questions about third-party usage, contact the project maintainers.
