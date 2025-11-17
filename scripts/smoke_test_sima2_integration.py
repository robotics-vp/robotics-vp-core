#!/usr/bin/env python3
"""
Smoke test for SIMA-2 stubs and context adapter.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.sima2_agent_stub import Sima2AgentStub
from src.sima2.sima2_context_adapter import sima2_rollout_to_decision_summary


def main():
    agent = Sima2AgentStub()
    rollout = agent.run_episode()
    summary = sima2_rollout_to_decision_summary(rollout)
    print("Rollout:", rollout)
    print("Decision summary:", summary)


if __name__ == "__main__":
    main()
