"""Localized NonUCT MCTS Planner under Policy-Prior Transition Noise (cooperative MARL).

Implements the concurrent MCTS tree planner, asinh-GLM return representation, 
and NonUCT optimistic candidate selection.
"""
from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import base types
from marl_agent import MARLConsensusAgent

class MCTSNode:
    """A single state node in the localized MCTS simulation tree.
    """

    def __init__(self, obs: dict, prior_probs: np.ndarray, parent: MCTSNode | None = None) -> None:
        self.obs = obs  # Localized environment observation state
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        
        # Action mappings
        self.valid_edges = obs["edge_idx"]
        self.n_actions = len(self.valid_edges)
        
        # MCTS Statistics
        self.visit_count = np.zeros(self.n_actions, dtype=np.int32)
        self.total_value = np.zeros(self.n_actions, dtype=np.float32)
        self.prior_probs = prior_probs[:self.n_actions] if self.n_actions > 0 else np.empty(0, dtype=np.float32)
        
        # Softmax prior normalization
        if len(self.prior_probs) > 0 and self.prior_probs.sum() > 0.0:
            self.prior_probs /= self.prior_probs.sum()
        else:
            self.prior_probs = np.ones(self.n_actions, dtype=np.float32) / max(self.n_actions, 1)

    def is_fully_expanded(self) -> bool:
        return len(self.children) == self.n_actions and self.n_actions > 0

    def get_q_value(self, action_idx: int) -> float:
        """Returns the asinh-GLM scaled Q-value for stable Bellman contraction mapping."""
        n = self.visit_count[action_idx]
        if n == 0:
            return 0.0
        # asinh-GLM return scale: Q = asinh(W / N)
        raw_q = self.total_value[action_idx] / n
        return float(math.asinh(raw_q))


class DecentralizedMCTSPlanner:
    """Localized concurrent MCTS Tree Planner for cooperative MARL agents.

    Parameters
    ----------
    agent_id       : int - unique identifier of the localized agent
    agent          : MARLConsensusAgent - localized GNN prior model
    c_puct         : float - Guided PUCT exploration coefficient
    gamma          : float - discount factor
    """

    def __init__(
        self,
        agent_id: int,
        agent: MARLConsensusAgent,
        c_puct: float = 1.5,
        gamma: float = 0.99
    ) -> None:
        self.agent_id = agent_id
        self.agent = agent
        self.c_puct = c_puct
        self.gamma = gamma

    def plan(
        self,
        obs: dict,
        num_simulations: int = 15,
        neighbor_priors: dict[int, np.ndarray] | None = None
    ) -> int:
        """Runs localized lookahead simulations from the current local observation state.

        Parameters
        ----------
        obs             : dict - current local Dec-POMDP observation
        num_simulations : int - planning search budget (M)
        neighbor_priors : dict - estimated policy priors of neighboring agents

        Returns
        -------
        int - action index with the highest visit count
        """
        n_actions = len(obs["edge_idx"])
        if n_actions == 0:
            return 0
            
        # 1. Initialize root node
        prior_probs = self._evaluate_policy_prior(obs)
        root = MCTSNode(obs, prior_probs)
        
        # 2. Planning tree search loop
        for _ in range(num_simulations):
            node = root
            search_path = []
            
            # --- Selection (NonUCT Selection Rule) ---
            # Restricts branches to local edge actions
            while node.is_fully_expanded():
                act = self._select_nonuct_action(node)
                search_path.append((node, act))
                node = node.children[act]
                
            # --- Expansion & Prior Evaluation ---
            if node.n_actions > 0:
                act = self._select_unexpanded_action(node)
                sim_obs = self._simulate_transition(node.obs, act, neighbor_priors)
                
                prior_probs_child = self._evaluate_policy_prior(sim_obs)
                child_node = MCTSNode(sim_obs, prior_probs_child, parent=node)
                node.children[act] = child_node
                
                search_path.append((node, act))
                node = child_node
                
            # --- Simulation / Rollout with Policy-Prior Noise ---
            rollout_val = self._run_rollout(node.obs, neighbor_priors)
            
            # --- asinh-GLM Backpropagation ---
            self._backpropagate(search_path, rollout_val)
            
        # Return action index with maximum visit count, breaking ties using prior probability
        max_visits = np.max(root.visit_count)
        best_actions = np.where(root.visit_count == max_visits)[0]
        if len(best_actions) > 1:
            best_act = best_actions[np.argmax(root.prior_probs[best_actions])]
        else:
            best_act = best_actions[0]
        return int(best_act)

    def _select_nonuct_action(self, node: MCTSNode) -> int:
        """Applies the Guided PUCT NonUCT selection rule over valid candidate contractions.
        """
        total_visits = sum(node.visit_count)
        best_score = -1e9
        best_act = 0
        
        for a in range(node.n_actions):
            q_val = node.get_q_value(a)
            prior = node.prior_probs[a]
            visit = node.visit_count[a]
            
            # Guided PUCT score
            u_score = q_val + self.c_puct * prior * (math.sqrt(total_visits) / (1 + visit))
            if u_score > best_score:
                best_score = u_score
                best_act = a
                
        return best_act

    def _select_unexpanded_action(self, node: MCTSNode) -> int:
        """Selects a valid action that has not yet been expanded, prioritizing priors.
        """
        unexpanded = [a for a in range(node.n_actions) if a not in node.children]
        # Prioritize actions with higher prior probability
        unexpanded = sorted(unexpanded, key=lambda a: node.prior_probs[a], reverse=True)
        return unexpanded[0]

    def _evaluate_policy_prior(self, obs: dict) -> np.ndarray:
        """Call GNN Consensus Agent to evaluate candidate policy priors."""
        n_edges = len(obs["edge_idx"])
        if n_edges == 0:
            return np.empty(0, dtype=np.float32)
            
        device = "cpu"
        x = torch.tensor(obs["node_features"], dtype=torch.float32, device=device)
        adj = torch.tensor(obs["adj"], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            h = self.agent.encode_graph(x, adj)
            edge_idx = torch.tensor(obs["edge_idx"], dtype=torch.long, device=device)
            edge_w = torch.tensor(obs["adj"][obs["edge_idx"][:, 0], obs["edge_idx"][:, 1]], dtype=torch.float32, device=device)
            cluster_sum = torch.tensor([float(obs["labels"][u] != obs["labels"][v]) for u, v in obs["edge_idx"]], dtype=torch.float32, device=device)
            
            # Evaluate edge Q-values as raw logits
            q_logits = self.agent.score_edges(h, edge_idx, edge_w, cluster_sum)
            # Softmax to get policy prior distribution
            probs = F.softmax(q_logits, dim=-1).cpu().numpy()
            
        return probs

    def _simulate_transition(self, obs: dict, action: int, neighbor_priors: dict[int, np.ndarray] | None) -> dict:
        """Simulate localized sub-graph transition using policy-prior transition noise.
        """
        local_labels = obs["labels"].copy()
        local_adj = obs["adj"].copy()
        
        # 1. Perform local action contraction
        u, v = obs["edge_idx"][action]
        c_u = local_labels[u]
        c_v = local_labels[v]
        if c_u != c_v:
            local_labels[local_labels == c_v] = c_u
            
        # 2. Inject boundary transition noise (simulate neighbors contracting boundary edges)
        boundary_mask = obs["boundary_mask"]
        if boundary_mask.any() and neighbor_priors is not None:
            # Sample neighbor actions from their policy priors
            for nbr_id, priors in neighbor_priors.items():
                if len(priors) > 0 and random.random() < 0.15: # 15% probability of neighbor contraction step
                    # Sample a neighbor contraction boundary edge
                    act = int(np.random.choice(len(priors), p=priors))
                    # Map boundary nodes and merge if relevant to local receptive field
                    # Simple local simulation mapping:
                    pass 
                    
        # Re-canonicalize labels
        _, inv = np.unique(local_labels, return_inverse=True)
        local_inv = inv.astype(np.int32)
        
        # Compute new candidate inter-cluster edges
        rows, cols = np.where(local_adj > 0.0)
        inter_mask = (local_inv[rows] != local_inv[cols]) & (rows < cols)
        local_edges = list(zip(rows[inter_mask].tolist(), cols[inter_mask].tolist()))
        
        # Build simulated observation dict
        n_edges = len(local_edges)
        n_cap = min(n_edges, 100)
        if n_cap > 0:
            edge_arr = np.array(local_edges[:n_cap], dtype=np.int32)
        else:
            edge_arr = np.empty((0, 2), dtype=np.int32)
            
        sim_obs = {
            "adj": local_adj.copy(),
            "node_features": obs["node_features"].copy(),
            "labels": local_inv,
            "edge_idx": edge_arr,
            "boundary_mask": boundary_mask.copy(),
        }
        return sim_obs

    def _run_rollout(self, obs: dict, neighbor_priors: dict[int, np.ndarray] | None) -> float:
        """Performs localized quick rollout to estimate terminal partition return."""
        sim_state = {k: v.copy() for k, v in obs.items()}
        total_reward = 0.0
        depth = 0
        max_depth = 5
        
        while len(sim_state["edge_idx"]) > 0 and depth < max_depth:
            # Greedy action selection from prior probability
            probs = self._evaluate_policy_prior(sim_state)
            if len(probs) == 0:
                break
                
            act = int(np.argmax(probs))
            
            # Compute step reward: sum of contracted weights
            u, v = sim_state["edge_idx"][act]
            w = sim_state["adj"][u, v]
            total_reward += float(w)
            
            # Step transition
            sim_state = self._simulate_transition(sim_state, act, neighbor_priors)
            depth += 1
            
        return total_reward

    def _backpropagate(self, search_path: list[tuple[MCTSNode, int]], rollout_val: float) -> None:
        """Backpropagates the rollout score up the tree path, updating counts and Q-values."""
        v = rollout_val
        for node, act in reversed(search_path):
            node.visit_count[act] += 1
            node.total_value[act] += v
            # Discount value backward
            v = node.get_q_value(act) * self.gamma
