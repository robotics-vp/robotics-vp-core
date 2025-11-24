"""
Inference module for running trained models in deployment/demo scenarios.
"""

from src.inference.demo_policy import DemoPolicy, DemoPolicyConfig

__all__ = ["DemoPolicy", "DemoPolicyConfig"]
