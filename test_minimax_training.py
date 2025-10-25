"""
Test script for minimax training data generation.

This script demonstrates how to generate training data using the minimax agent
for bootstrapping neural network training.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer, generate_minimax_training_data
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent


def test_minimax_data_generation():
    """Test minimax training data generation."""
    print("=== Testing Minimax Training Data Generation ===\n")
    
    # Create trainer
    trainer = Trainer()
    
    # Generate training data using minimax
    print("Generating training data using minimax agent...")
    policy_data, value_data = trainer.generate_minimax_training_data(
        num_games=20,  # Small number for testing
        minimax_depth=3  # Shallow depth for speed
    )
    
    print(f"\nGenerated {len(policy_data)} policy training examples")
    print(f"Generated {len(value_data)} value training examples")
    
    # Analyze the data
    print("\n=== Data Analysis ===")
    
    # Look at a few examples
    print("\nFirst 3 policy examples:")
    for i in range(min(3, len(policy_data))):
        board, action_probs = policy_data[i]
        chosen_action = np.argmax(action_probs)
        print(f"  Example {i+1}: Chosen action = {chosen_action}, Max prob = {action_probs[chosen_action]:.3f}")
    
    print("\nFirst 3 value examples:")
    for i in range(min(3, len(value_data))):
        board, value = value_data[i]
        print(f"  Example {i+1}: Board evaluation = {value:.3f}")
    
    # Value distribution
    values = [item[1] for item in value_data]
    print(f"\nValue distribution:")
    print(f"  Min: {min(values):.3f}")
    print(f"  Max: {max(values):.3f}")
    print(f"  Mean: {np.mean(values):.3f}")
    print(f"  Wins (+1): {sum(1 for v in values if v > 0.5)}")
    print(f"  Losses (-1): {sum(1 for v in values if v < -0.5)}")
    print(f"  Neutral: {sum(1 for v in values if -0.5 <= v <= 0.5)}")
    
    return policy_data, value_data


def test_minimax_vs_random():
    """Test minimax agent performance against random."""
    print("\n=== Testing Minimax vs Random Performance ===\n")
    
    # Generate more games to see performance
    game_records = generate_minimax_training_data(num_games=50, minimax_depth=4)
    
    # Count minimax wins
    minimax_wins = 0
    random_wins = 0
    ties = 0
    
    for game in game_records:
        winner = game['winner']
        minimax_is_player1 = game['minimax_is_player1']
        
        if winner == 0:
            ties += 1
        elif (winner == 1 and minimax_is_player1) or (winner == -1 and not minimax_is_player1):
            minimax_wins += 1
        else:
            random_wins += 1
    
    total_games = len(game_records)
    print(f"Results from {total_games} games:")
    print(f"  Minimax wins: {minimax_wins} ({100*minimax_wins/total_games:.1f}%)")
    print(f"  Random wins: {random_wins} ({100*random_wins/total_games:.1f}%)")
    print(f"  Ties: {ties} ({100*ties/total_games:.1f}%)")
    
    # Average game length
    avg_length = np.mean([game['num_moves'] for game in game_records])
    print(f"  Average game length: {avg_length:.1f} moves")


def test_network_training():
    """Test training networks with minimax data."""
    print("\n=== Testing Network Training with Minimax Data ===\n")
    
    # Create networks
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    
    # Create trainer
    trainer = Trainer()
    
    # Generate training data
    print("Generating training data...")
    policy_data, value_data = trainer.generate_minimax_training_data(
        num_games=30,
        minimax_depth=3
    )
    
    # Train policy network
    print("\nTraining policy network...")
    policy_results = trainer.train_policy_network(
        policy_net,
        policy_data,
        num_epochs=10,  # Small number for testing
        batch_size=16
    )
    print(f"Policy network final loss: {policy_results['final_loss']:.6f}")
    
    # Train value network
    print("\nTraining value network...")
    value_results = trainer.train_value_network(
        value_net,
        value_data,
        num_epochs=10,  # Small number for testing
        batch_size=16
    )
    print(f"Value network final loss: {value_results['final_loss']:.6f}")
    
    print("\nNetworks trained successfully!")


if __name__ == "__main__":
    print("Testing Minimax Training Data Generation\n")
    
    try:
        # Test 1: Basic data generation
        policy_data, value_data = test_minimax_data_generation()
        
        # Test 2: Performance analysis
        test_minimax_vs_random()
        
        # Test 3: Network training
        test_network_training()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()