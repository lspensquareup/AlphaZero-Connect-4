# AlphaZero Connect-4 Environment Setup

## Conda Environment: chess-gym

This project uses the existing `chess-gym` conda environment with additional packages for AlphaZero implementation.

### Environment Information
- **Environment Name**: chess-gym
- **Python Version**: 3.11.14
- **Location**: `/opt/homebrew/Caskroom/miniforge/base/envs/chess-gym`

### Key Dependencies

#### Core ML/Game Libraries
- `pytorch=2.2.2` - Neural network framework
- `torchvision=0.17.2` - Computer vision utilities
- `numpy=1.26.4` - Numerical computing (downgraded for PyTorch compatibility)
- `gymnasium=1.2.1` - Reinforcement learning environments

#### Visualization and Progress
- `matplotlib=3.10.7` - Plotting and visualization
- `tqdm=4.67.1` - Progress bars

#### Game-specific
- `chess=1.11.2` - Chess library (already installed)

### Setup Instructions

1. **Activate the environment**:
   ```bash
   conda activate chess-gym
   ```

2. **Verify installation**:
   ```bash
   python -c "import torch, numpy, gymnasium, matplotlib; print('All dependencies available')"
   ```

3. **Run training**:
   ```bash
   python train_with_minimax.py
   ```

### Project Structure

```
AlphaZero-Connect-4/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Abstract base agent
│   ├── policy_agent.py    # Neural network agent
│   └── mcts_agent.py      # MCTS-enhanced agent
├── networks/              # Neural network architectures
│   ├── policy_network.py  # Action prediction network
│   └── value_network.py   # Position evaluation network
├── training/              # Training infrastructure
│   ├── trainer.py         # Main training logic with minimax
│   └── tournament.py      # Agent evaluation
├── connect4_env.py        # Gymnasium Connect-4 environment
├── train_with_minimax.py  # Complete training pipeline
├── test_minimax_simple.py # Basic functionality tests
└── models/                # Saved model checkpoints
```

### Usage Examples

#### Basic Training Data Generation
```python
from training.trainer import generate_minimax_training_data

# Generate training games
game_records = generate_minimax_training_data(
    num_games=100,
    minimax_depth=4
)
```

#### Neural Network Training
```python
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork

trainer = Trainer()
policy_net = PolicyNetwork()
value_net = ValueNetwork()

# Generate data and train
policy_data, value_data = trainer.generate_minimax_training_data(50, 4)
trainer.train_policy_network(policy_net, policy_data, num_epochs=20)
trainer.train_value_network(value_net, value_data, num_epochs=20)
```

### Model Output

Trained models are saved in the `models/` directory:
- `policy_network_epoch_N.pth` - Policy network checkpoints
- `value_network_epoch_N.pth` - Value network checkpoints
- `training_session.json` - Training statistics and metadata

### Next Steps

1. **Enhanced Agents**: Use trained networks in PolicyAgent for gameplay
2. **MCTS Integration**: Combine neural networks with Monte Carlo Tree Search
3. **Self-Play Training**: Implement full AlphaZero self-play loop
4. **Evaluation**: Tournament system for comparing different agents

### Troubleshooting

**NumPy Compatibility**: If you encounter NumPy/PyTorch compatibility issues:
```bash
conda install "numpy<2" -y
```

**Memory Issues**: For large training runs, reduce batch size or number of games:
```python
# Reduce memory usage
trainer.generate_minimax_training_data(num_games=20, minimax_depth=3)
trainer.train_policy_network(net, data, batch_size=8)
```

**CUDA**: To use GPU acceleration (if available):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```