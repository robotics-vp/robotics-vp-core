#!/usr/bin/env python3
"""
Print objective presets and weights for quick inspection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.objective_profile import get_objective_presets


def main():
    presets = get_objective_presets()
    print("Objective presets:")
    for name, vec in presets.items():
        weights = vec.to_list() if hasattr(vec, "to_list") else vec
        print(f"  {name}: {weights}")


if __name__ == "__main__":
    main()
