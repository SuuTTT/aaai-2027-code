# Decentralized Multi-Agent Graph Partitioning Fleet (AAAI 2027)

Official PyTorch implementation of the AAAI 2027 paper: **"Decentralized Multi-Agent Graph Partitioning Fleet via Event-Triggered Quantized Gumbel-Softmax GAT Consensus"**.

This repository contains the high-performance, resource-efficient cooperative MARL framework for solving massive signed multicut partitioning (MCMP) and community detection tasks. By decomposing global graphs into localized overlapping seed-based receptive fields, our framework enforces a strict $O(1)$ VRAM footprint (running in under 50 MB VRAM) while achieving state-of-the-art zero-shot scale generalization.

---

## 🚀 Key Scientific Frameworks
1. **Seed-Based Receptive Fields (`marl_env.py`)**: Slices massive graphs into localized overlapping subgraphs ($|V_i| \le 35$) via bounded BFS hops, maintaining constant memory complexity at any scale.
2. **Event-Triggered Gumbel-Softmax GAT (`marl_agent.py`)**: Uses Gumbel-Softmax categorical projection to compress continuous node states into 16-bit discrete keys ($512\times$ bandwidth compression). Suppresses physical packet transmission using linear recurrence state estimators.
3. **Prior-Guided NonUCT MCTS Planner (`marl_mcts.py`)**: Eliminates tree search tie-breaking failures under small search budgets by guiding visit-count argmax selections with GNN softmax priors.
4. **Global Cluster-Sum termination (`marl_mcts.py` & `marl_env.py`)**: Resolves truncation bias by checking global multicut rewards in $O(1)$ to terminate contractions exactly when rewards turn negative.

---

## 📂 Codebase Layout

```
.
├── marl_env.py          # Dec-POMDP multi-agent graph environments and local credit rewards
├── marl_agent.py        # Gumbel-Softmax discrete quantizers, GAT consensus layers, and estimators
├── marl_mcts.py         # Prior-guided NonUCT tree-search planners and cluster coordination
├── train_marl_fleet.py  # IQL training loop and zero-shot scale generalization sweeps
├── tests/               # Validation suite covering state transitions and action masks
└── .gitignore           # LaTeX and checkpoint ignore templates
```

---

## 🛠️ Installation & Setup

Configure your Python environment and install core dependencies:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu # or cu118 for GPU support
pip install networkx numpy scipy pytest
```

---

## 📈 Quickstart: Training & Evaluation

### A. Train the Cooperative Fleet
Train the decentralized GAT agents using Independent Q-Learning (IQL) on small-scale training graphs ($N=20$):
```bash
python train_marl_fleet.py --episodes 100 --scale 20 --lr 0.001
```

### B. Inductive Zero-Shot Scale Generalization Sweep
Evaluate the trained model zero-shot on larger out-of-distribution scales ($N \in \{40, 100, 200\}$) on random (ER) and scale-free (BA) topologies:
```bash
python train_marl_fleet.py --eval --checkpoint results/checkpoints/marl_fleet_best.pt --scales 40 100 200
```

---

## 🧪 Running Unit Tests
Verify structural correctness, state transitions, and contract-action masking boundaries by running our test suite:
```bash
pytest tests/
```

---

## 📄 Citation

```bibtex
@inproceedings{aaai2027marlfleet,
  title={Decentralized Multi-Agent Graph Partitioning Fleet via Event-Triggered Quantized Gumbel-Softmax GAT Consensus},
  author={AAAI Submission 2842},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2027}
}
```
