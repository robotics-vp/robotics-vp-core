#!/usr/bin/env python3
"""
Dependency hierarchy smoke test.

Asserts:
- Lower-level modules (envs, rl, diffusion, sim, robot backends) do NOT import EconomicController/DatapackEngine.
- EconomicController/DatapackEngine do NOT import semantic/vla/diffusion.
- SemanticOrchestrator (if present) may import econ/datapacks, but not vice versa.
"""
import os
import sys
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_forbidden_imports(forbidden_modules, search_dirs):
    violations = []
    for d in search_dirs:
        full = os.path.join(ROOT, d)
        paths = []
        if os.path.isdir(full):
            for root, _, files in os.walk(full):
                for fname in files:
                    if fname.endswith(".py"):
                        paths.append(os.path.join(root, fname))
        elif os.path.isfile(full):
            paths.append(full)
        for path in paths:
            with open(path, "r") as f:
                try:
                    tree = ast.parse(f.read(), filename=path)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for forb in forbidden_modules:
                        if mod.startswith(forb):
                            violations.append((path, mod))
                if isinstance(node, ast.Import):
                    for n in node.names:
                        for forb in forbidden_modules:
                            if n.name.startswith(forb):
                                violations.append((path, n.name))
    return violations


def main():
    forbidden = [
        "src.orchestrator.economic_controller",
        "src.orchestrator.datapack_engine",
    ]
    lower_dirs = [
        "src/envs",
        "src/rl",
        "src/diffusion",
        "src/sima",
        "src/robot",
        "src/vla",
    ]
    violations = scan_forbidden_imports(forbidden, lower_dirs)
    if violations:
        print("FORBIDDEN IMPORTS DETECTED:")
        for path, mod in violations:
            print(f"  {path} imports {mod}")
        sys.exit(1)

    # Ensure econ controller/engine don't import semantic/vla/diffusion
    econ_dirs = [
        "src/orchestrator/economic_controller.py",
        "src/orchestrator/datapack_engine.py",
    ]
    forbidden_up = ["src.vla", "src.sima", "src.diffusion", "src.robot"]
    violations = scan_forbidden_imports(forbidden_up, econ_dirs)
    if violations:
        print("Econ modules importing higher-level components (not allowed):")
        for path, mod in violations:
            print(f"  {path} imports {mod}")
        sys.exit(1)

    print("Dependency hierarchy check passed.")


if __name__ == "__main__":
    main()
