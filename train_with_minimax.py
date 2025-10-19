"""
Example script demonstrating the complete AlphaZero Connect-4 training pipeline
using conda environment and minimax-generated training data.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from connect4_env import GymnasiumConnectFour


def main():
    """Run the complete training pipeline."""
    print("=== AlphaZero Connect-4 Training Pipeline ===\n")
    
    # Check environment
    print("Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current working directory: {os.getcwd()}")
    print()
    
    # Create trainer and networks
    print("1. Initializing networks and trainer...")
    trainer = Trainer(save_dir="models")
    
    policy_net = PolicyNetwork(hidden_size=128)
    value_net = ValueNetwork(hidden_size=128)
    
    print(f"Policy network parameters: {policy_net.get_num_parameters():,}")
    print(f"Value network parameters: {value_net.get_num_parameters():,}")
    print()
    
    # Generate training data using minimax
    print("2. Generating training data using minimax agent...")
    print("This may take a few minutes...")
    
    policy_data, value_data = trainer.generate_minimax_training_data(
        num_games=50,  # Generate 50 games
        minimax_depth=4  # Use depth 4 for stronger play
    )
    
    print(f"Generated {len(policy_data)} training examples")
    print()
    
    # Analyze training data
    print("3. Analyzing training data quality...")
    values = [item[1] for item in value_data]
    wins = sum(1 for v in values if v > 0.5)
    losses = sum(1 for v in values if v < -0.5)
    neutral = len(values) - wins - losses
    
    print(f"Value distribution: {wins} wins, {losses} losses, {neutral} neutral positions")
    print(f"Value range: [{min(values):.3f}, {max(values):.3f}]")
    print()
    
    # Train policy network
    print("4. Training policy network...")
    policy_results = trainer.train_policy_network(
        policy_net,
        policy_data,
        num_epochs=20,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"Policy training completed. Final loss: {policy_results['final_loss']:.6f}")
    print()
    
    # Train value network
    print("5. Training value network...")
    value_results = trainer.train_value_network(
        value_net,
        value_data,
        num_epochs=20,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"Value training completed. Final loss: {value_results['final_loss']:.6f}")
    print()
    
    # Test trained networks
    print("6. Testing trained networks...")
    test_trained_networks(policy_net, value_net)
    
    # Save training statistics
    trainer.training_stats.update({
        'policy_final_loss': policy_results['final_loss'],
        'value_final_loss': value_results['final_loss'],
        'training_examples': len(policy_data),
        'minimax_games': 50
    })
    
    trainer.save_training_stats('training_session.json')
    
    print("\n=== Training Pipeline Complete! ===")
    print(f"Models saved in: {trainer.save_dir}/")
    print("Next steps:")
    print("1. Use these networks to create PolicyAgent")
    print("2. Implement MCTS for enhanced gameplay")
    print("3. Start self-play training loop")


def test_trained_networks(policy_net, value_net):
    """Test the trained networks on sample positions."""
    env = GymnasiumConnectFour()
    env.reset()
    
    print("Testing on empty board:")
    
    # Test policy network
    policy_net.eval()
    with torch.no_grad():
        board_tensor = torch.FloatTensor(env.board).unsqueeze(0)
        policy_logits = policy_net(board_tensor)
        policy_probs = torch.softmax(policy_logits, dim=1)
        
        print(f"Policy probabilities: {policy_probs.squeeze().numpy()}")
        best_action = torch.argmax(policy_probs).item()
        print(f"Recommended action: column {best_action}")
    
    # Test value network  
    value_net.eval()
    with torch.no_grad():
        value_estimate = value_net(board_tensor)
        print(f"Position value: {value_estimate.item():.3f}")
    
    print()


if __name__ == "__main__":
    main()