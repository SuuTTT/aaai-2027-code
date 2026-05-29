"""Localized Multi-Agent Graph Partitioning Environment (Dec-POMDP MARL).

Decomposes a large global graph into localized, overlapping seed-based receptive 
fields (ego-networks) and executes concurrent local edge contractions.
"""
from __future__ import annotations

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

# Import base types from single-agent codebase for compatibility
from rlgb.tasks.base import Problem

MAX_EDGES = 100  # Local max branching actions per agent


def slice_graph_receptive_fields(
    adj: np.ndarray, num_agents: int = 5, hop_k: int = 2, max_nodes: int = 40
) -> list[np.ndarray]:
    """Slice a global signed graph into overlapping seed-based receptive fields.

    Parameters
    ----------
    adj        : np.ndarray - global signed adjacency matrix (N x N)
    num_agents : int - number of local agents (M)
    hop_k      : int - BFS hop expansion limit
    max_nodes  : int - maximum node capacity constraint per receptive field

    Returns
    -------
    list[np.ndarray] - List of 1D node index arrays representing the receptive field
                       subgraph node mapping for each agent.
    """
    n = adj.shape[0]
    # Simple absolute weights to identify high-degree structural seed nodes
    abs_adj = np.abs(adj)
    degrees = abs_adj.sum(axis=1)
    
    # Select seeds: starts with highest degree nodes, enforcing minimum separation
    seeds: list[int] = []
    candidates = np.argsort(degrees)[::-1].tolist()
    
    for cand in candidates:
        if len(seeds) >= num_agents:
            break
        # Ensure seed separation (not directly connected if possible)
        is_separated = True
        for s in seeds:
            if abs_adj[s, cand] > 0.0:
                is_separated = False
                break
        if is_separated or len(seeds) == 0:
            seeds.append(cand)
            
    # Fallback to pure top-degree if not enough separated seeds found
    if len(seeds) < num_agents:
        for cand in candidates:
            if len(seeds) >= num_agents:
                break
            if cand not in seeds:
                seeds.append(cand)

    receptive_fields = []
    
    # Perform localized BFS hop expansion around each seed
    for seed in seeds:
        visited = {seed}
        queue = [(seed, 0)]
        head = 0
        while head < len(queue):
            curr, dist = queue[head]
            head += 1
            if dist >= hop_k or len(visited) >= max_nodes:
                continue
            # Expand neighbors
            neighbors = np.where(abs_adj[curr] > 0.0)[0]
            for nbr in neighbors:
                nbr = int(nbr)
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
                    if len(visited) >= max_nodes:
                        break
        receptive_fields.append(np.array(sorted(list(visited)), dtype=np.int32))

    return receptive_fields


class DecentralizedGraphEnv:
    """Decentralized Multi-Agent Graph Contraction Environment (Dec-POMDP).

    Parameters
    ----------
    problem    : Problem - containing global graph G and cost matrix
    num_agents : int - number of active localized agents (M)
    hop_k      : int - BFS hops for local subgraphs
    max_nodes  : int - max capacity per sub-graph
    beta       : float - scaling coefficient for cooperative boundary penalty
    horizon    : int - episode length limit before truncation
    """

    def __init__(
        self,
        problem: Problem,
        num_agents: int = 4,
        hop_k: int = 2,
        max_nodes: int = 35,
        beta: float = 0.15,
        horizon: int = 40,
    ) -> None:
        self.problem = problem
        self.global_adj = problem.adj.copy()
        self.n_global = self.global_adj.shape[0]
        self.num_agents = num_agents
        self.hop_k = hop_k
        self.max_nodes = max_nodes
        self.beta = beta
        self.horizon = horizon
        self._step_count = 0
        
        # Decompose global graph
        self.node_maps = slice_graph_receptive_fields(
            self.global_adj, num_agents=self.num_agents, hop_k=self.hop_k, max_nodes=self.max_nodes
        )
        
        # Track active boundaries
        self.boundary_masks = []
        for i, mapping in enumerate(self.node_maps):
            # Node is on boundary if it is shared with ANY other agent
            mask = np.zeros(len(mapping), dtype=bool)
            for idx, node in enumerate(mapping):
                shared = False
                for j, other_map in enumerate(self.node_maps):
                    if i != j and node in other_map:
                        shared = True
                        break
                mask[idx] = shared
            self.boundary_masks.append(mask)
            
        # Action space per agent
        self.action_space = spaces.Discrete(MAX_EDGES)

    def reset(self, seed: int | None = None) -> dict[int, dict]:
        """Reset Dec-POMDP state and return localized observations per agent."""
        self._step_count = 0
        
        # Global label state: start in agglomerative singleton format
        self.global_labels = np.arange(self.n_global, dtype=np.int32)
        
        # Compute local observations
        return self._build_observations()

    def step(self, actions: dict[int, int]) -> tuple[
        dict[int, dict],            # observations per agent
        dict[int, float],           # local credit-assigned rewards
        dict[int, bool],            # agent termination flags
        dict[int, bool],            # agent truncation flags
        dict                        # debug info
    ]:
        """Execute concurrent local contractions across agents."""
        old_global_labels = self.global_labels.copy()
        
        # 1. Execute contractions concurrently per agent
        for agent_id, act in actions.items():
            node_map = self.node_maps[agent_id]
            local_labels = self.global_labels[node_map]
            
            # Find candidate local inter-cluster edges
            local_adj = self.global_adj[np.ix_(node_map, node_map)]
            rows, cols = np.where(local_adj > 0.0)
            inter_mask = (local_labels[rows] != local_labels[cols]) & (rows < cols)
            local_edges = list(zip(rows[inter_mask].tolist(), cols[inter_mask].tolist()))
            
            n_edges = len(local_edges)
            if n_edges > 0:
                edge_idx = int(act) % n_edges
                u_local, v_local = local_edges[edge_idx]
                
                # Map to global indices
                u_global = int(node_map[u_local])
                v_global = int(node_map[v_local])
                
                c_u = self.global_labels[u_global]
                c_v = self.global_labels[v_global]
                
                if c_u != c_v:
                    # Concurrently update global labels
                    self.global_labels[self.global_labels == c_v] = c_u

        # 2. Canonicalize global labels to maintain sequential indexing
        uniq, inv = np.unique(self.global_labels, return_inverse=True)
        self.global_labels = inv.astype(np.int32)
        
        self._step_count += 1
        
        # 3. Compute credit-assigned rewards & local termination
        obs_dict = self._build_observations()
        rewards = {}
        terminated = {}
        truncated = {}
        
        for i in range(self.num_agents):
            node_map = self.node_maps[i]
            local_adj = self.global_adj[np.ix_(node_map, node_map)]
            
            # Local Credit Reward = Internal delta - beta * boundary cut penalty
            # A edge (u, v) is cut if labels are different
            old_local_labels = old_global_labels[node_map]
            new_local_labels = self.global_labels[node_map]
            
            # Compute internal objective change: sum of contracted positive edges
            internal_r = 0.0
            rows, cols = np.where(local_adj != 0.0)
            for u_l, v_l in zip(rows, cols):
                if u_l < v_l:
                    w = local_adj[u_l, v_l]
                    # If previously separate but now merged
                    was_separate = old_local_labels[u_l] != old_local_labels[v_l]
                    is_merged = new_local_labels[u_l] == new_local_labels[v_l]
                    
                    if was_separate and is_merged:
                        internal_r += w # positive reward for merging attractive edges, negative for repulsive
                        
            # Compute boundary cut penalty: penalizes cutting positive weights along boundaries
            boundary_penalty = 0.0
            boundary_mask = self.boundary_masks[i]
            for u_l, v_l in zip(rows, cols):
                if u_l < v_l:
                    # Check if either endpoint lies on shared boundary
                    if boundary_mask[u_l] or boundary_mask[v_l]:
                        w = local_adj[u_l, v_l]
                        is_cut = new_local_labels[u_l] != new_local_labels[v_l]
                        # Penalize cutting positive boundary edges
                        if is_cut and w > 0.0:
                            boundary_penalty += w
                            
            rewards[i] = float(internal_r - self.beta * boundary_penalty)
            
            # Check localized termination: terminated if no positive inter-cluster edges remain,
            # OR if all positive candidate edges have a non-positive global cluster sum (GAEC reward <= 0).
            n_local_edges = obs_dict[i]["n_edges"][0]
            local_terminated = (n_local_edges == 0)
            if not local_terminated:
                o = obs_dict[i]
                filtered = o["edge_idx"]
                cost_adj = self.problem.meta["cost_matrix"]
                
                max_sum = -9999.0
                seen = {}
                for idx in range(len(filtered)):
                    u_l, v_l = int(filtered[idx, 0]), int(filtered[idx, 1])
                    u_g = int(node_map[u_l])
                    v_g = int(node_map[v_l])
                    cu_g, cv_g = self.global_labels[u_g], self.global_labels[v_g]
                    key = (min(cu_g, cv_g), max(cu_g, cv_g))
                    if key not in seen:
                        # True global cluster sum
                        mask_u_global = self.global_labels == cu_g
                        mask_v_global = self.global_labels == cv_g
                        seen[key] = float(cost_adj[np.ix_(mask_u_global, mask_v_global)].sum())
                    max_sum = max(max_sum, seen[key])
                
                if max_sum <= 0.0:
                    local_terminated = True
            terminated[i] = local_terminated
            truncated[i] = self._step_count >= self.horizon

        # Global termination is logical AND of agent terminations
        global_term = all(terminated.values())
        global_trunc = self._step_count >= self.horizon
        
        info = {
            "global_labels": self.global_labels,
            "k_current": int(np.unique(self.global_labels).shape[0]),
            "global_terminated": global_term,
            "global_truncated": global_trunc,
        }
        
        return obs_dict, rewards, terminated, truncated, info

    def _build_observations(self) -> dict[int, dict]:
        """Construct local Dec-POMDP observations for all agents."""
        obs_dict = {}
        for i in range(self.num_agents):
            node_map = self.node_maps[i]
            local_labels = self.global_labels[node_map]
            local_adj = self.global_adj[np.ix_(node_map, node_map)]
            
            # Canonicalize local labels for spatial representation invariance
            _, local_inv = np.unique(local_labels, return_inverse=True)
            local_inv = local_inv.astype(np.int32)
            
            # Find candidate positive inter-cluster edges in local subgraph
            rows, cols = np.where(local_adj > 0.0)
            inter_mask = (local_inv[rows] != local_inv[cols]) & (rows < cols)
            local_edges = list(zip(rows[inter_mask].tolist(), cols[inter_mask].tolist()))
            
            cost_adj = self.problem.meta["cost_matrix"]
            pos_edges = []
            for u_l, v_l in local_edges:
                u_g = int(node_map[u_l])
                v_g = int(node_map[v_l])
                if cost_adj[u_g, v_g] > 0.0:
                    pos_edges.append((u_l, v_l))
                    
            n_edges = len(pos_edges)
            n_cap = min(n_edges, MAX_EDGES)
            if n_cap > 0:
                edge_arr = np.array(pos_edges[:n_cap], dtype=np.int32)
            else:
                edge_arr = np.empty((0, 2), dtype=np.int32)
                
            # Node features (degree, boundary indicator, and label-count)
            n_local = len(node_map)
            k_local = int(local_inv.max()) + 1
            node_feats = np.zeros((n_local, 3), dtype=np.float32)
            node_feats[:, 0] = np.abs(local_adj).sum(axis=1) / max(1.0, float(n_local)) # normalized degree
            node_feats[:, 1] = self.boundary_masks[i].astype(np.float32) # boundary node indicator
            node_feats[:, 2] = np.array([float(np.sum(local_inv == c)) for c in local_inv]) / max(1.0, float(n_local)) # normalized cluster sizes
            
            obs_dict[i] = {
                "adj": local_adj.copy(),
                "node_features": node_feats,
                "labels": local_inv,
                "edge_idx": edge_arr,
                "n_edges": np.array([n_edges], dtype=np.int32),
                "boundary_mask": self.boundary_masks[i].copy(),
                "node_map": node_map.copy(), # global node indices for coordinate projection
            }
        return obs_dict
