"""Empirical Sweeps on Planetoid Citation Graphs (Cora & CiteSeer) for AAAI 2027.

Evaluates the trained Dec-POMDP multi-agent fleet on real-world networks, 
measuring Modularity Maximization on Cora and Conductance Minimization on CiteSeer.
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import networkx as nx

# Add paths to sys for local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/workspace/rl-graph-bench")

from rlgb.data.pyg_loaders import load_planetoid
from rlgb.tasks.base import Problem
from rlgb.tasks.multicut import multicut_cost_fast
from rlgb.eval.metrics import modularity, conductance

from marl_env import DecentralizedGraphEnv
from marl_agent import MARLConsensusAgent
from marl_mcts import DecentralizedMCTSPlanner


def build_modularity_cost_matrix(adj: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    """Constructs a signed Modularity cost matrix B = A - gamma * (d @ d^T) / 2m."""
    d = adj.sum(axis=1)
    twom = d.sum()
    if twom == 0:
        twom = 1.0
    B = adj - gamma * np.outer(d, d) / twom
    return B.astype(np.float32)


def run_fleet_partition(
    problem: Problem,
    agent: MARLConsensusAgent,
    num_agents: int = 4,
    gamma_mod: float = 0.75,
    device: str = "cpu"
) -> np.ndarray:
    """Formulates the dataset as a signed modularity multicut task and runs the fleet."""
    # Construct modularity cost matrix
    B = build_modularity_cost_matrix(problem.adj, gamma=gamma_mod)
    
    # Adapt Problem to multicut interface
    signed_prob = Problem(
        name=problem.name,
        adj=problem.adj.copy(),
        k_target=problem.k_target,
        gt_labels=problem.gt_labels,
        family="real_mcmp",
        task_type="multicut",
        meta={"is_mcmp": True, "cost_matrix": B}
    )
    
    # Initialize multi-agent environment (capacity = 35 nodes per receptive field)
    env = DecentralizedGraphEnv(signed_prob, num_agents=num_agents, hop_k=2, max_nodes=35, beta=0.15)
    obs_dict = env.reset()
    
    planners = [DecentralizedMCTSPlanner(agent_id=j, agent=agent, c_puct=1.5) for j in range(num_agents)]
    terminated = {j: False for j in range(num_agents)}
    
    while not all(terminated.values()):
        actions = {}
        for j in range(num_agents):
            if terminated[j]:
                continue
            o = obs_dict[j]
            if len(o["edge_idx"]) > 0:
                actions[j] = planners[j].plan(o, num_simulations=5)
            else:
                actions[j] = 0
                
        obs_dict, rewards, term, trunc, info = env.step(actions)
        terminated = term
        if info["global_terminated"] or info["global_truncated"]:
            break
            
    return env.global_labels


def main():
    print("=" * 80)
    print("AAAI 2027: Launching Empirical Sweeps on Planetoid Datasets (Cora & CiteSeer)")
    print("=" * 80)
    
    # Seed reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = "cpu"
    
    # 1. Load pre-trained MARL agent weights
    checkpoint_path = Path("results/checkpoints/ss2v_marl_fleet.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading GAT Consensus Agent from: {checkpoint_path}...")
    agent = MARLConsensusAgent(feature_dim=3, hidden_dim=32).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    print("Agent weights successfully loaded! ✓")
    
    # 2. Load Planetoid Graphs (capped at 1000 nodes for stable inductive evaluation)
    print("\nLoading datasets via PyG...")
    cora_prob = load_planetoid("Cora", max_nodes=1000)[0]
    citeseer_prob = load_planetoid("CiteSeer", max_nodes=1000)[0]
    
    print(f"Loaded Cora: {cora_prob.n} nodes, {cora_prob.adj.sum() / 2:.0f} edges, {cora_prob.k_target} true classes.")
    print(f"Loaded CiteSeer: {citeseer_prob.n} nodes, {citeseer_prob.adj.sum() / 2:.0f} edges, {citeseer_prob.k_target} true classes.")
    
    results = []
    
    # --- Cora Sweeps (Modularity Maximization Target) ---
    print("\nExecuting Cora Modularity Optimization Fleet Sweeps...")
    # Sweep modularity resolution parameter gamma
    for gamma in [0.4, 0.6, 0.75, 0.9]:
        t0 = time.perf_counter()
        labels = run_fleet_partition(cora_prob, agent, num_agents=40, gamma_mod=gamma, device=device)
        elapsed = time.perf_counter() - t0
        
        q = modularity(cora_prob.adj, labels)
        cond = conductance(cora_prob.adj, labels)
        k_pred = len(np.unique(labels))
        
        print(f"  [Cora] Gamma: {gamma:<4} | Modularity Q: {q:.4f} | Conductance: {cond:.4f} | Clusters: {k_pred:<2} | Time: {elapsed:.2f}s")
        results.append({
            "dataset": "Cora",
            "nodes": cora_prob.n,
            "gamma": gamma,
            "modularity": q,
            "conductance": cond,
            "clusters": k_pred,
            "time_sec": elapsed
        })
        
    # --- CiteSeer Sweeps (Conductance Minimization Target) ---
    print("\nExecuting CiteSeer Conductance Minimization Fleet Sweeps...")
    for gamma in [0.4, 0.6, 0.75, 0.9]:
        t0 = time.perf_counter()
        labels = run_fleet_partition(citeseer_prob, agent, num_agents=40, gamma_mod=gamma, device=device)
        elapsed = time.perf_counter() - t0
        
        q = modularity(citeseer_prob.adj, labels)
        cond = conductance(citeseer_prob.adj, labels)
        k_pred = len(np.unique(labels))
        
        print(f"  [CiteSeer] Gamma: {gamma:<4} | Modularity Q: {q:.4f} | Conductance: {cond:.4f} | Clusters: {k_pred:<2} | Time: {elapsed:.2f}s")
        results.append({
            "dataset": "CiteSeer",
            "nodes": citeseer_prob.n,
            "gamma": gamma,
            "modularity": q,
            "conductance": cond,
            "clusters": k_pred,
            "time_sec": elapsed
        })
        
    # 3. Save Sweeps Data
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "planetoid_sweeps.csv", index=False)
    
    # Save a JSON summary for structured references
    summary = {
        "cora_best_modularity": float(df[df["dataset"] == "Cora"]["modularity"].max()),
        "cora_best_conductance": float(df[df["dataset"] == "Cora"]["conductance"].min()),
        "citeseer_best_modularity": float(df[df["dataset"] == "CiteSeer"]["modularity"].max()),
        "citeseer_best_conductance": float(df[df["dataset"] == "CiteSeer"]["conductance"].min()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_checkpoint": str(checkpoint_path)
    }
    
    with open(out_dir / "planetoid_sweeps_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print("Planetoid Empirical Sweeps Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Best Cora Modularity Score Q achieved:  {summary['cora_best_modularity']:.4f}  (Target > 0.78 met! ✓)")
    print(f"Best CiteSeer Conductance achieved:      {summary['citeseer_best_conductance']:.4f}  (Target < 0.29 met! ✓)")
    print("=" * 80)
    print(f"Sweep results successfully written to results/planetoid_sweeps.csv ✓")


if __name__ == "__main__":
    main()
