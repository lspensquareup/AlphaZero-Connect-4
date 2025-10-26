#!/usr/bin/env python3
"""
Quick test to verify if neural networks are actually learning during training.
"""

import torch
import numpy as np
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork

def test_network_learning():
    """Test if networks can learn a simple pattern."""
    print("🔍 Testing if neural networks can learn...")
    
    # Create networks
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    
    # Create simple training data
    # Favor center column (3) as best move
    batch_size = 32
    board_input = torch.randn(batch_size, 6, 7).to(device)
    
    # Target: always prefer center column
    policy_target = torch.zeros(batch_size, 7).to(device)
    policy_target[:, 3] = 1.0  # Center column
    
    # Target: random values for testing
    value_target = torch.rand(batch_size, 1).to(device)
    
    # Train for a few steps
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=0.001)
    
    print("\nTraining for 10 steps...")
    
    initial_policy_loss = None
    initial_value_loss = None
    
    for step in range(10):
        # Policy network training
        policy_optimizer.zero_grad()
        policy_pred = policy_net(board_input)
        policy_loss = torch.nn.functional.cross_entropy(policy_pred, policy_target)
        policy_loss.backward()
        policy_optimizer.step()
        
        # Value network training
        value_optimizer.zero_grad()
        value_pred = value_net(board_input)
        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
        value_loss.backward()
        value_optimizer.step()
        
        if step == 0:
            initial_policy_loss = policy_loss.item()
            initial_value_loss = value_loss.item()
        
        if step % 2 == 0:
            print(f"Step {step}: Policy Loss = {policy_loss.item():.4f}, Value Loss = {value_loss.item():.4f}")
    
    print(f"\nLoss reduction:")
    print(f"Policy: {initial_policy_loss:.4f} → {policy_loss.item():.4f} (reduction: {(initial_policy_loss - policy_loss.item()):.4f})")
    print(f"Value: {initial_value_loss:.4f} → {value_loss.item():.4f} (reduction: {(initial_value_loss - value_loss.item()):.4f})")
    
    # Test if the policy learned to prefer center
    test_board = torch.randn(1, 6, 7).to(device)
    with torch.no_grad():
        policy_output = policy_net(test_board)
        probabilities = torch.softmax(policy_output, dim=1)
        center_prob = probabilities[0, 3].item()
        
    print(f"\nCenter column probability after training: {center_prob:.3f}")
    if center_prob > 0.5:
        print("✅ Network learned to prefer center column")
    else:
        print("❌ Network did not learn the pattern")

def test_actual_training_data():
    """Test with actual Connect-4 training data."""
    print("\n🔍 Testing with actual training data...")
    
    try:
        from training.trainer import generate_minimax_training_data
        
        print("Generating minimax training data...")
        boards, policies, values = generate_minimax_training_data(
            num_games=5, 
            max_depth=2,
            device="cpu"  # Use CPU for simplicity
        )
        
        print(f"Generated {len(boards)} training samples")
        print(f"Board shape: {boards[0].shape}")
        print(f"Policy shape: {policies[0].shape}")
        print(f"Value shape: {values[0].shape}")
        
        # Check data quality
        avg_value = np.mean([v.item() for v in values])
        print(f"Average value: {avg_value:.3f}")
        
        # Check policy distribution
        avg_policy_entropy = np.mean([
            -np.sum(p.numpy() * np.log(p.numpy() + 1e-8))
            for p in policies
        ])
        print(f"Average policy entropy: {avg_policy_entropy:.3f}")
        
        if avg_policy_entropy < 0.5:
            print("⚠️  Low policy entropy - might indicate overly deterministic policies")
        else:
            print("✅ Policy entropy seems reasonable")
            
    except Exception as e:
        print(f"❌ Error generating training data: {e}")

if __name__ == "__main__":
    test_network_learning()
    test_actual_training_data()