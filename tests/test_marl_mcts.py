#!/usr/bin/env python3
"""Unit tests for localized concurrent NonUCT MCTS Planner (marl_mcts.py).
"""
import os
import sys
import torch
import numpy as np
import math
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/iclr-2027-code")

from rlgb.data.mcmp_instances import er_mcmp
from marl_agent import MARLConsensusAgent
from marl_mcts import DecentralizedMCTSPlanner, MCTSNode

def test_asinh_glm_node_q_values():
    print("Testing asinh-GLM value scale projection...")
    obs = {
        "edge_idx": np.array([[0, 1], [1, 2]], dtype=np.int32),
        "labels": np.array([0, 1, 2], dtype=np.int32),
        "adj": np.eye(3, dtype=np.float32)
    }
    priors = np.array([0.5, 0.5], dtype=np.float32)
    
    node = MCTSNode(obs, priors)
    assert node.n_actions == 2, "Action mapping length mismatch"
    
    # Manually backpropagate some dummy values
    node.visit_count[0] = 5
    node.total_value[0] = 10.0  # Mean Q = 2.0
    
    # Q should be asinh(2.0) = 1.4436
    q_val = node.get_q_value(0)
    expected_q = math.asinh(2.0)
    assert np.isclose(q_val, expected_q), f"asinh-GLM Q value mismatch, expected {expected_q}, got {q_val}"
    
    print(f"  asinh-GLM scaled Q value: {q_val:.4f} (Raw mean: 2.0)")
    print("asinh-GLM node Q-value test passed! ✓\n")

def test_mcts_planner_e2e():
    print("Testing DecentralizedMCTSPlanner end-to-end planning sweep...")
    # Load 40-node ER graph
    er_probs = er_mcmp(n=40, n_instances=1, seed_offset=42)
    p = er_probs[0]
    
    # Initialize GNN Prior agent
    agent = MARLConsensusAgent(feature_dim=3, hidden_dim=32)
    agent.eval()
    
    # Initialize local planner for Agent 0
    planner = DecentralizedMCTSPlanner(agent_id=0, agent=agent, c_puct=1.5, gamma=0.99)
    
    # Build a mock 10-node local sub-graph observation
    n_nodes = 10
    local_adj = p.meta["cost_matrix"][:n_nodes, :n_nodes]
    local_labels = np.arange(n_nodes, dtype=np.int32)
    
    node_feats = np.zeros((n_nodes, 3), dtype=np.float32)
    node_feats[:, 0] = np.abs(local_adj).sum(axis=1) # degree
    node_feats[:, 1] = np.zeros(n_nodes, dtype=np.float32) # boundary
    node_feats[:, 2] = np.ones(n_nodes, dtype=np.float32) # cluster size
    
    edge_idx = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[4] = True  # shared boundary node
    
    obs = {
        "adj": local_adj,
        "node_features": node_feats,
        "labels": local_labels,
        "edge_idx": edge_idx,
        "boundary_mask": boundary_mask
    }
    
    # Mock neighbor priors (e.g. Neighbor 1 has 5 candidate actions)
    neighbor_priors = {
        1: np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    }
    
    # Execute 15 planning simulations (the ICLR default budget)
    best_act = planner.plan(obs, num_simulations=15, neighbor_priors=neighbor_priors)
    
    assert 0 <= best_act < len(edge_idx), f"Planned action index {best_act} is out of valid bounds"
    print(f"  MCTS Planned Action Index: {best_act} (contracting edge {edge_idx[best_act].tolist()})")
    
    print("Decentralized MCTS planner end-to-end sweep passed! ✓\n")

if __name__ == "__main__":
    test_asinh_glm_node_q_values()
    test_mcts_planner_e2e()
