#!/usr/bin/env python3
"""Unit tests for localized multi-agent graph partitioning environment (marl_env.py).
"""
import os
import sys
import numpy as np
from pathlib import Path

# Insert project root to path for both single-agent and new repositories
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/iclr-2027-code")

from rlgb.data.mcmp_instances import er_mcmp
from marl_env import DecentralizedGraphEnv, slice_graph_receptive_fields

def test_graph_slicing():
    print("Testing graph receptive field slicing...")
    # Load 40-node ER graph
    er_probs = er_mcmp(n=40, n_instances=1, seed_offset=42)
    p = er_probs[0]
    adj = p.meta["cost_matrix"]
    
    node_maps = slice_graph_receptive_fields(adj, num_agents=4, hop_k=2, max_nodes=20)
    
    assert len(node_maps) == 4, "Should slice exactly 4 receptive fields"
    for i, mapping in enumerate(node_maps):
        assert len(mapping) > 0, f"Agent {i} receptive field should not be empty"
        assert len(mapping) <= 20, f"Agent {i} field should respect max_nodes cap"
        print(f"  Agent {i} field: {len(mapping)} nodes center-sliced.")
        
    print("Graph slicing test passed! ✓\n")

def test_marl_env_stepping():
    print("Testing DecentralizedGraphEnv step transitions & rewards...")
    # Load 40-node ER graph
    er_probs = er_mcmp(n=40, n_instances=1, seed_offset=42)
    p = er_probs[0]
    
    env = DecentralizedGraphEnv(p, num_agents=4, hop_k=2, max_nodes=20, beta=0.15)
    obs = env.reset()
    
    for i in range(4):
        assert i in obs, f"Agent {i} observation missing on reset"
        o = obs[i]
        assert "adj" in o, "local adj missing"
        assert "node_features" in o, "node_features missing"
        assert "labels" in o, "local labels missing"
        assert "edge_idx" in o, "edge_idx missing"
        assert "n_edges" in o, "n_edges missing"
        assert "boundary_mask" in o, "boundary_mask missing"
        assert len(o["node_features"].shape) == 2, "features should be 2D matrix"
        assert o["node_features"].shape[1] == 3, "features dimension should be 3 (degree, boundary, cluster-size)"
        
    # Execute dummy joint actions
    actions = {0: 0, 1: 0, 2: 0, 3: 0} # contract the first valid inter-cluster edge per agent
    next_obs, rewards, term, trunc, info = env.step(actions)
    
    assert len(next_obs) == 4, "Observation list shape mismatch"
    assert len(rewards) == 4, "Reward list shape mismatch"
    assert len(term) == 4, "Terminated dict mismatch"
    
    print("  Joint rewards computed:")
    for i, r in rewards.items():
        assert isinstance(r, float), "rewards must be float scalar values"
        print(f"    Agent {i} credit-assigned reward: {r:.4f}")
        
    print("  Global partition status after step:")
    print(f"    Total clusters remaining: {info['k_current']}")
    print(f"    Global termination status: {info['global_terminated']}")
    
    print("MARL environment step and reward tests passed! ✓\n")

if __name__ == "__main__":
    test_graph_slicing()
    test_marl_env_stepping()
