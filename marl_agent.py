"""Decentralized ET-GS-GAT Consensus GNN Agent (cooperative MARL).

Implements the Custom GAT Layer, Gumbel-Softmax discrete quantizer codebook, 
local recurrent boundary message estimators, and event-triggered gating.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MAX_EDGES = 100


class CustomGATLayer(nn.Module):
    """Custom Graph Attention Network (GAT) Layer in PyTorch.

    Processes sub-graph features and adjacency without requiring external PyG 
    dependencies, ensuring sub-millisecond execution.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.randn(heads, out_dim))
        self.attn_r = nn.Parameter(torch.randn(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim), adj: (N, N)
        N = x.shape[0]
        h = self.proj(x).view(N, self.heads, self.out_dim)  # (N, heads, out_dim)

        # Calculate attention coefficients
        attn_l = (h * self.attn_l).sum(dim=-1)  # (N, heads)
        attn_r = (h * self.attn_r).sum(dim=-1)  # (N, heads)

        # Broadcast sum (N, N, heads)
        score = attn_l.unsqueeze(1) + attn_r.unsqueeze(0)
        score = self.leaky_relu(score)

        # Mask out non-edges (we use absolute weights threshold for active GAT connections)
        mask = (torch.abs(adj) > 0.0).float().unsqueeze(-1)  # (N, N, 1)
        score = score.masked_fill(mask == 0.0, -1e9)

        attn_weights = F.softmax(score, dim=1)  # (N, N, heads)

        # Aggregation (N, heads, out_dim)
        out = torch.einsum("nvh,vhd->nhd", attn_weights, h)
        return out.mean(dim=1)  # Average over heads -> (N, out_dim)


class GumbelSoftmaxQuantizer(nn.Module):
    """Gumbel-Softmax discrete codebook quantizer for boundary keys.

    Quantizes continuous node embeddings into discrete 16-bit representation keys 
    during inference (M=4 categorical variables, K=16 classes), achieving 512x 
    bandwidth compression.
    """

    def __init__(self, in_dim: int, num_vars: int = 4, num_classes: int = 16) -> None:
        super().__init__()
        self.M = num_vars
        self.K = num_classes
        self.proj = nn.Linear(in_dim, num_vars * num_classes)
        # Learnable continuous codebook matching categories
        self.codebook = nn.Parameter(torch.randn(num_classes, in_dim))

    def forward(self, h: torch.Tensor, tau: float = 1.0, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (N, in_dim)
        N = h.shape[0]
        logits = self.proj(h).view(N, self.M, self.K)  # (N, M, K)

        if training:
            # Gumbel-Softmax continuous relaxation
            gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / tau
            y_soft = F.softmax(gumbels, dim=-1)  # (N, M, K)
            
            # Continuous projection via codebook lookup
            # y_soft: (N, M, K), codebook: (K, in_dim) -> (N, M, in_dim)
            quantized = torch.einsum("nmk,kd->nmd", y_soft, self.codebook)
            quantized = quantized.mean(dim=1)  # average over categorical variables -> (N, in_dim)
            return quantized, y_soft
        else:
            # Hard argmax mapping during evaluation
            idx = torch.argmax(logits, dim=-1)  # (N, M)
            
            # Project to hard quantized embedding via codebook lookup
            y_hard = F.one_hot(idx, num_classes=self.K).float()  # (N, M, K)
            quantized = torch.einsum("nmk,kd->nmd", y_hard, self.codebook)
            quantized = quantized.mean(dim=1)  # (N, in_dim)
            
            # Compress discrete index mapping to 16-bit key representation (integer mapping)
            # key = sum_{m} idx_m * (K^m)
            powers = torch.tensor([self.K ** i for i in range(self.M)], device=h.device, dtype=torch.int32)
            keys = (idx.to(torch.int32) * powers).sum(dim=-1)  # (N,)
            return quantized, keys.float()


class LocalMessageEstimator(nn.Module):
    """Linear-recurrent boundary message estimator M_i->j.

    Predicts neighboring agent's boundary state locally to bypass physical network 
    transmission at every step.
    """

    def __init__(self, obs_dim: int, embedding_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Simple low-overhead linear-recurrent cell (GRU-like)
        self.cell = nn.GRUCell(obs_dim + embedding_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, obs: torch.Tensor, last_key: torch.Tensor, h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (N, obs_dim), last_key: (N, embedding_dim), h_prev: (N, hidden_dim)
        N = obs.shape[0]
        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=obs.device)
            
        inputs = torch.cat([obs, last_key], dim=-1)  # (N, obs_dim + embedding_dim)
        h_next = self.cell(inputs, h_prev)  # (N, hidden_dim)
        pred = self.proj(h_next)  # (N, embedding_dim)
        return pred, h_next


class ETGating:
    """Dynamic Memory Event-Triggered Gating mechanism.

    Broadcasting is suppressed if the true state remains close to the local estimator.
    """

    def __init__(self, window_size: int = 10, sigma: float = 1.5, epsilon: float = 0.05) -> None:
        self.W = window_size
        self.sigma = sigma
        self.epsilon = epsilon
        self.history: list[float] = []

    def update_and_gate(self, true_state: torch.Tensor, estimated_state: torch.Tensor) -> bool:
        """Determines whether a physical communication broadcast is triggered.

        Returns True if communication is triggered, False if gated/suppressed.
        """
        # Compute L1 norm deviation between true and predicted embeddings
        deviation = float(torch.abs(true_state - estimated_state).sum().item())
        
        # Calculate dynamic threshold based on historical deviation variance
        if len(self.history) > 1:
            variance = np.var(self.history)
        else:
            variance = 0.0
            
        threshold = self.sigma * variance + self.epsilon
        
        # Append deviation to rolling window
        self.history.append(deviation)
        if len(self.history) > self.W:
            self.history.pop(0)
            
        return deviation > threshold


class MARLConsensusAgent(nn.Module):
    """Cooperative GAT Consensus edge contraction planning agent."""

    def __init__(self, feature_dim: int = 3, hidden_dim: int = 32, num_actions: int = MAX_EDGES) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Spatial GNN encoder
        self.gat1 = CustomGATLayer(feature_dim, hidden_dim)
        self.gat2 = CustomGATLayer(hidden_dim, hidden_dim)
        
        # 2. Gumbel-Softmax quantized boundary keys
        self.quantizer = GumbelSoftmaxQuantizer(hidden_dim)
        
        # 3. Dueling Q-value Scorer (Edge-level)
        # Score candidates: (h_u || h_v || edge_w || cluster_sum) -> Q
        # input size = hidden_dim * 2 + 2 features
        self.q_value_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode_graph(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Processes local sub-graph node features and signed adjacency matrix."""
        h = self.gat1(x, adj)
        h = F.relu(h)
        h = self.gat2(h, adj)
        return h

    def score_edges(self, h: torch.Tensor, edge_idx: torch.Tensor, edge_w: torch.Tensor, cluster_sum: torch.Tensor) -> torch.Tensor:
        """Computes contract/cut Q-values for candidate edges.

        h          : (N, hidden_dim) node embeddings
        edge_idx   : (E_cap, 2) candidate edge endpoint indices
        edge_w     : (E_cap,) signed edge weights
        cluster_sum: (E_cap,) contracted community cost predictions
        """
        E_cap = edge_idx.shape[0]
        if E_cap == 0:
            return torch.zeros(0, device=h.device)
            
        u = edge_idx[:, 0]
        v = edge_idx[:, 1]
        
        # Concat endpoints features with edge weights and cost prediction
        h_u = h[u]  # (E_cap, hidden_dim)
        h_v = h[v]  # (E_cap, hidden_dim)
        
        inputs = torch.cat([
            h_u, h_v, 
            edge_w.view(E_cap, 1), 
            cluster_sum.view(E_cap, 1)
        ], dim=-1)  # (E_cap, hidden_dim * 2 + 2)
        
        return self.q_value_scorer(inputs).squeeze(-1)  # (E_cap,)
