#!/usr/bin/env python3
"""
Debug script to test neural network training an    results_trained = trainer.evaluate_agent_vs_baseline(
        trained_agent, minimax_agent, num_games=20  # Larger sample size
    )valuation.
"""

import torch
import numpy as np
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from agents.policy_agent import PolicyAgent
from agents.minimax_agent import MinimaxAgent

def main():
    print("🔍 Debugging Training and Evaluation...")
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create networks
    policy_net = PolicyNetwork(hidden_size=128).to(device)
    trainer = Trainer()
    
    print("\n1. Testing untrained network vs Minimax...")
    # Test untrained network
    untrained_agent = PolicyAgent(name="Untrained")
    untrained_agent.set_network(policy_net.eval())
    untrained_agent.temperature = 0.1  # Low randomness
    
    minimax_agent = MinimaxAgent(depth=2)
    
    results_untrained = trainer.evaluate_agent_vs_baseline(
        untrained_agent, minimax_agent, num_games=20  # Larger sample size
    )
    print(f"Untrained vs Minimax-2: {results_untrained['win_rate']:.1%}")
    
    print("\n2. Generating training data...")
    # Generate substantial training data
    policy_data, value_data = trainer.generate_minimax_training_data(
        num_games=20, minimax_depth=3  # More realistic parameters
    )
    print(f"Generated {len(policy_data)} training examples")
    
    # Check data quality
    if len(policy_data) > 0:
        board, target = policy_data[0]
        print(f"Sample policy target shape: {target.shape}")
        print(f"Sample policy target sum: {np.sum(target):.3f}")
        print(f"Sample policy target max: {np.max(target):.3f}")
    
    print("\n3. Training the network...")
    # Get initial weights
    initial_param = next(policy_net.parameters()).data.clone()
    
    # Train the network
    policy_net.train()
    results = trainer.train_policy_network(
        policy_net, policy_data, num_epochs=50, batch_size=16, device=device  # More epochs
    )
    
    # Check weight changes
    final_param = next(policy_net.parameters()).data
    weight_change = torch.norm(final_param - initial_param).item()
    
    print(f"Training complete - Loss: {results['final_loss']:.6f}")
    print(f"Weight change magnitude: {weight_change:.6f}")
    
    print("\n4. Testing trained network vs Minimax...")
    # Test trained network
    trained_agent = PolicyAgent(name="Trained")
    trained_agent.set_network(policy_net.eval())
    trained_agent.temperature = 0.1  # Low randomness
    
    results_trained = trainer.evaluate_agent_vs_baseline(
        trained_agent, minimax_agent, num_games=5  # Fewer games for speed
    )
    print(f"Trained vs Minimax-2: {results_trained['win_rate']:.1%}")
    
    print("\n5. Comparison:")
    print(f"Untrained: {results_untrained['win_rate']:.1%}")
    print(f"Trained:   {results_trained['win_rate']:.1%}")
    
    improvement = results_trained['win_rate'] - results_untrained['win_rate']
    print(f"Improvement: {improvement:+.1%}")
    
    if improvement > 0.1:
        print("✅ Training appears to be working!")
    elif improvement > 0.05:
        print("⚠️ Slight improvement, may need more training")
    else:
        print("❌ No significant improvement - potential issue")

if __name__ == "__main__":
    main()