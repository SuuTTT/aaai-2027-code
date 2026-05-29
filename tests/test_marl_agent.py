#!/usr/bin/env python3
"""Unit tests for Gumbel-Softmax consensus GAT Agent (marl_agent.py).
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from marl_agent import (
    MARLConsensusAgent, 
    GumbelSoftmaxQuantizer, 
    LocalMessageEstimator, 
    ETGating
)

def test_gumbel_softmax_quantization():
    print("Testing Gumbel-Softmax Quantization & Key Compression...")
    quantizer = GumbelSoftmaxQuantizer(in_dim=32, num_vars=4, num_classes=16)
    
    # 5 dummy boundary node embeddings
    h = torch.randn(5, 32)
    
    # Test training phase continuous relaxation
    q_soft, y_soft = quantizer(h, tau=1.0, training=True)
    assert q_soft.shape == (5, 32), "Quantized soft embedding shape mismatch"
    assert y_soft.shape == (5, 4, 16), "Logits probabilities shape mismatch"
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones(5, 4)), "Soft probabilities must sum to 1.0"
    
    # Test evaluation phase hard quantization
    q_hard, keys = quantizer(h, training=False)
    assert q_hard.shape == (5, 32), "Quantized hard embedding shape mismatch"
    assert keys.shape == (5,), "Quantized key array shape mismatch"
    for k in keys.tolist():
        assert 0 <= k < 65536, f"Keys must fit in standard 16-bit space (0-65535), got {k}"
        
    print("  Quantized keys successfully mapped:")
    print(f"    Sample 16-bit discrete key keys: {keys.tolist()}")
    print("Gumbel-Softmax quantization test passed! ✓\n")

def test_local_message_estimator():
    print("Testing Local Boundary Message Estimator...")
    # obs dim = 3, embedding dim = 32, GRU hidden dim = 16
    estimator = LocalMessageEstimator(obs_dim=3, embedding_dim=32, hidden_dim=16)
    
    obs = torch.randn(5, 3)
    last_key = torch.randn(5, 32)
    
    pred_1, h_next_1 = estimator(obs, last_key)
    assert pred_1.shape == (5, 32), "Message prediction shape mismatch"
    assert h_next_1.shape == (5, 16), "GRU hidden state shape mismatch"
    
    # Step again to verify temporal recurrence
    pred_2, h_next_2 = estimator(obs, pred_1, h_next_1)
    assert pred_2.shape == (5, 32), "Step 2 prediction shape mismatch"
    assert not torch.allclose(pred_1, pred_2), "Recurrent state must change over step transitions"
    
    print("Message estimator recurrence test passed! ✓\n")

def test_event_triggered_gating():
    print("Testing Dynamic Memory Event-Triggered Gating...")
    gating = ETGating(window_size=5, sigma=1.5, epsilon=0.1)
    
    # True and estimated states stay close
    true_1 = torch.zeros(32)
    est_1 = torch.zeros(32)
    # The first updates initialize the threshold variance
    trigger = gating.update_and_gate(true_1, est_1)
    assert not trigger, "Gating should suppress communication when states match perfectly"
    
    for _ in range(5):
        # Small changes should keep communication gated/suppressed
        true_noise = torch.randn(32) * 0.01
        gating.update_and_gate(true_noise, est_1)
        
    # Introduce a massive state change (e.g. edge contraction occurs)
    true_jump = torch.ones(32) * 5.0
    trigger = gating.update_and_gate(true_jump, est_1)
    assert trigger, "Event trigger must fire upon a sudden structural boundary shift"
    
    print("Event-triggered communication gating test passed! ✓\n")

def test_agent_forward_pass():
    print("Testing MARLConsensusAgent forward pass & edge scoring...")
    agent = MARLConsensusAgent(feature_dim=3, hidden_dim=32)
    agent.eval()
    
    # 5 nodes sub-graph features (degree, boundary, cluster-size)
    x = torch.randn(5, 3)
    # Fully-connected signed cost matrix
    adj = torch.tensor([
        [0.0, 1.0, -0.5, 0.0, 0.0],
        [1.0, 0.0, 1.0, -0.2, 0.0],
        [-0.5, 1.0, 0.0, 0.5, 1.0],
        [0.0, -0.2, 0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    h = agent.encode_graph(x, adj)
    assert h.shape == (5, 32), "Encoded graph embedding shape mismatch"
    
    # Candidate contractions: contract edge (0,1), (1,2), (2,3) -> 3 candidate edges
    edge_idx = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    edge_w = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    cluster_sum = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float32)
    
    q_values = agent.score_edges(h, edge_idx, edge_w, cluster_sum)
    assert q_values.shape == (3,), "Q-value scorer output shape mismatch"
    assert not torch.isnan(q_values).any(), "Q-values contain NaN"
    
    print(f"  Agent Q-values computed successfully: {q_values.tolist()}")
    print("MARL consensus agent forward test passed! ✓\n")

if __name__ == "__main__":
    test_gumbel_softmax_quantization()
    test_local_message_estimator()
    test_event_triggered_gating()
    test_agent_forward_pass()
