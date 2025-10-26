#!/usr/bin/env python3
"""
Focused debug script to investigate network training issues.
"""

import torch
import numpy as np
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from agents.policy_agent import PolicyAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def debug_training_quality():
    """Debug the actual training process."""
    print("🔬 Debugging Training Quality...")
    
    # Get device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using MPS")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Create fresh networks
    policy_net = PolicyNetwork().to(device)
    trainer = Trainer()
    
    # Test 1: Check if training data generation works
    print("\n📚 Test 1: Training Data Generation")
    try:
        policy_data, value_data = trainer.generate_minimax_training_data(
            num_games=5, minimax_depth=2
        )
        print(f"✅ Generated {len(policy_data)} policy examples, {len(value_data)} value examples")
        
        # Check data quality
        if len(policy_data) > 0:
            sample_board, sample_policy = policy_data[0]
            print(f"Sample policy target sum: {np.sum(sample_policy):.3f}")
            print(f"Sample policy target: {sample_policy}")
        
    except Exception as e:
        print(f"❌ Training data generation failed: {e}")
        return
    
    # Test 2: Check initial network prediction
    print("\n🧠 Test 2: Initial Network Predictions")
    sample_board = np.zeros((6, 7))  # Empty board
    board_tensor = torch.FloatTensor(sample_board).unsqueeze(0).to(device)
    
    with torch.no_grad():
        initial_pred = policy_net(board_tensor)
        print(f"Initial prediction shape: {initial_pred.shape}")
        print(f"Initial prediction: {initial_pred.cpu().numpy()[0]}")
        print(f"Initial prediction sum: {initial_pred.sum().item():.3f}")
    
    # Test 3: Check training actually changes weights
    print("\n🏋️ Test 3: Weight Change During Training")
    
    # Save initial weights
    initial_weights = policy_net.state_dict()['fc1.weight'].clone()
    initial_norm = torch.norm(initial_weights).item()
    print(f"Initial weight norm: {initial_norm:.6f}")
    
    # Train the network
    print("Training for 1 epoch...")
    results = trainer.train_policy_network(
        policy_net, policy_data, num_epochs=1, batch_size=16, learning_rate=0.001, device=device
    )
    print(f"Training loss: {results['final_loss']:.6f}")
    
    # Check weight changes
    final_weights = policy_net.state_dict()['fc1.weight']
    final_norm = torch.norm(final_weights).item()
    weight_change = torch.norm(final_weights - initial_weights).item()
    
    print(f"Final weight norm: {final_norm:.6f}")
    print(f"Weight change magnitude: {weight_change:.6f}")
    
    if weight_change < 1e-6:
        print("❌ CRITICAL: Weights barely changed! Training not working.")
    else:
        print("✅ Weights changed appropriately.")
    
    # Test 4: Check prediction change
    print("\n🎯 Test 4: Prediction Change After Training")
    with torch.no_grad():
        final_pred = policy_net(board_tensor)
        pred_change = torch.norm(final_pred - initial_pred).item()
        print(f"Final prediction: {final_pred.cpu().numpy()[0]}")
        print(f"Prediction change magnitude: {pred_change:.6f}")
    
    # Test 5: Quick evaluation test
    print("\n⚔️ Test 5: Agent Performance Before/After Training")
    
    # Test before training (using initial weights)
    policy_net_before = PolicyNetwork().to(device)  # Fresh network
    agent_before = PolicyAgent("Before Training")
    agent_before.set_network(policy_net_before.eval())
    agent_before.temperature = 0.1
    
    # Test after training
    agent_after = PolicyAgent("After Training")
    agent_after.set_network(policy_net.eval())
    agent_after.temperature = 0.1
    
    # Evaluate vs random
    random_agent = RandomAgent("Random")
    
    print("Testing untrained agent vs random...")
    result_before = trainer.evaluate_agent_vs_baseline(agent_before, random_agent, num_games=10)
    print(f"Before training: {result_before['win_rate']:.1%} win rate")
    
    print("Testing trained agent vs random...")
    result_after = trainer.evaluate_agent_vs_baseline(agent_after, random_agent, num_games=10)
    print(f"After training: {result_after['win_rate']:.1%} win rate")
    
    improvement = result_after['win_rate'] - result_before['win_rate']
    print(f"Improvement: {improvement:.1%}")
    
    if improvement > 0.1:  # 10% improvement
        print("✅ Training shows positive improvement!")
    elif improvement > -0.1:  # Within 10%
        print("⚠️ Training shows minimal change (could be noise)")
    else:
        print("❌ Training made performance worse!")
    
    # Test 6: MCTS evaluation
    print("\n🌳 Test 6: MCTS Performance Check")
    mcts_agent = MCTSAgent(simulations=50, name="MCTS-50")  # Weak MCTS
    
    print("Testing trained agent vs weak MCTS...")
    result_vs_mcts = trainer.evaluate_agent_vs_baseline(agent_after, mcts_agent, num_games=10)
    print(f"Trained agent vs MCTS-50: {result_vs_mcts['win_rate']:.1%} win rate")
    
    if result_vs_mcts['win_rate'] > 0.2:  # 20%+ vs weak MCTS
        print("✅ Reasonable performance vs MCTS")
    else:
        print("❌ Very poor performance vs MCTS")

if __name__ == "__main__":
    debug_training_quality()