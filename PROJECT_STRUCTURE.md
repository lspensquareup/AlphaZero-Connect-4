# AlphaZero Connect-4 Project Structure

This project implements Connect-4 with both traditional rule-based AI and modern neural network agents.

## 📁 Project Structure

```
/AlphaZero-Connect-4/
├── # EXISTING GAME FILES
├── connect4_env.py              # ✅ Game environment (Gymnasium compatible)
├── connect4_gui.py              # ✅ GUI interface  
├── run_connect4_gui.py          # ✅ GUI launcher
├── connect4_ai.py               # ✅ Rule-based AI agents
│
├── # NEW AI TRAINING SYSTEM
├── agents/                      # 🆕 AI agent implementations
│   ├── __init__.py
│   ├── base_agent.py            # Base class for all agents
│   ├── policy_agent.py          # Neural network policy agent
│   ├── value_agent.py           # Neural network value agent
│   └── random_agent.py          # Random baseline agent
│
├── networks/                    # 🆕 Neural network architectures
│   ├── __init__.py
│   ├── policy_network.py        # PyTorch policy network
│   └── value_network.py         # PyTorch value network
│
├── training/                    # 🆕 Training and evaluation
│   ├── __init__.py
│   ├── tournament.py            # AI vs AI tournaments
│   └── trainer.py               # Training loops
│
├── models/                      # 🆕 Saved neural networks
├── data/                        # 🆕 Training data and results
└── run_tournament.py            # 🆕 Tournament launcher
```

## 🚀 Quick Start

### 1. Run Your Existing GUI
```bash
python run_connect4_gui.py
```

### 2. Test AI Tournament System
```bash
# Quick test with rule-based agents
python run_tournament.py --mode quick

# Full baseline tournament  
python run_tournament.py --mode baseline --games 20

# Head-to-head match
python run_tournament.py --mode head2head --agent1 "Minimax" --agent2 "Greedy" --games 10
```

### 3. Create Sample Neural Networks (Optional)
```bash
python run_tournament.py --mode create-networks
```

## 🤖 Available Agents

### Rule-Based Agents
- **RandomAgent**: Makes random valid moves
- **CenterAgent**: Prefers center columns
- **GreedyAgent**: One-move lookahead with win/block logic
- **MinimaxAgent**: Minimax with alpha-beta pruning (configurable depth)

### Neural Network Agents
- **PolicyAgent**: Uses neural network to predict action probabilities
- **ValueAgent**: Uses neural network to evaluate positions + minimax
- Agents can load trained models from the `models/` directory

## 🔧 Dependencies

### Required (for rule-based agents)
- `numpy`
- `gymnasium` (for the Connect-4 environment)

### Optional (for neural networks)
- `torch` (PyTorch)
- `torchvision`

Install PyTorch:
```bash
pip install torch torchvision
```

## 📊 Tournament System

The tournament system provides:
- **Round-robin tournaments**: Every agent plays every other agent
- **Head-to-head matches**: Specific agent matchups
- **Statistical tracking**: Win rates, points, detailed match history
- **Results saving**: Tournament results saved as JSON files

## 🧠 Neural Network Training

The framework is set up for:
- **Policy networks**: Learn to predict good moves
- **Value networks**: Learn to evaluate board positions
- **Self-play training**: Future implementation of AlphaZero-style training
- **Tournament evaluation**: Compare different network versions

## 🎯 Next Steps

1. **Try the existing functionality**:
   ```bash
   python run_tournament.py --mode quick
   ```

2. **Install PyTorch for neural networks**:
   ```bash
   pip install torch torchvision
   ```

3. **Create sample networks**:
   ```bash
   python run_tournament.py --mode create-networks
   ```

4. **Run neural vs baseline tournament**:
   ```bash
   python run_tournament.py --mode neural
   ```

5. **Implement training data collection** (future work)
6. **Add MCTS for AlphaZero-style training** (future work)

## 🔍 File Details

- **agents/base_agent.py**: Abstract base class with common interface
- **networks/policy_network.py**: CNN for action probability prediction
- **networks/value_network.py**: CNN for position evaluation
- **training/tournament.py**: Tournament management and statistics
- **training/trainer.py**: Neural network training utilities
- **connect4_ai.py**: Traditional rule-based AI implementations

Your existing files (`connect4_env.py`, `connect4_gui.py`, etc.) remain unchanged and fully functional!