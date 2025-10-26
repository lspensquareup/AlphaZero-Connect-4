#!/usr/bin/env python3
"""
Debug script to investigate MCTS evaluation issues.
"""

import torch
import numpy as np
from agents.policy_agent import PolicyAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from networks.policy_network import PolicyNetwork
from training.trainer import Trainer

def debug_mcts_issue():
    """Debug the MCTS evaluation issue systematically."""
    print("🔍 Debugging MCTS evaluation issue...")
    
    # Create agents
    policy_agent = PolicyAgent("DebugPolicy")
    mcts_agent = MCTSAgent(simulations=100, name="DebugMCTS")
    random_agent = RandomAgent("DebugRandom")
    trainer = Trainer()
    
    # Test 1: Untrained network vs Random (should be ~50%)
    print("\n📊 Test 1: Untrained Policy vs Random")
    result = trainer.evaluate_agent_vs_baseline(policy_agent, random_agent, num_games=20)
    print(f"Result: {result['win_rate']:.1%} ({result['wins']}W-{result['losses']}L-{result['ties']}T)")
    
    # Test 2: Untrained network vs MCTS (should be low but not 0%)
    print("\n📊 Test 2: Untrained Policy vs MCTS-100")
    result = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_agent, num_games=20)
    print(f"Result: {result['win_rate']:.1%} ({result['wins']}W-{result['losses']}L-{result['ties']}T)")
    
    # Test 3: MCTS vs Random (should be high, like 80%+)
    print("\n📊 Test 3: MCTS-100 vs Random")
    result = trainer.evaluate_agent_vs_baseline(mcts_agent, random_agent, num_games=20)
    print(f"Result: {result['win_rate']:.1%} ({result['wins']}W-{result['losses']}L-{result['ties']}T)")
    
    # Test 4: Check if policy agent is making reasonable moves
    print("\n🧠 Test 4: Policy Agent Move Analysis")
    from connect4_env import GymnasiumConnectFour
    env = GymnasiumConnectFour()
    state = env.reset()[0]
    
    # Test policy agent moves
    policy_agent.player_id = 1
    valid_actions = [i for i in range(7) if state[0][i] == 0]  # Check top row for valid moves
    
    print(f"Valid actions: {valid_actions}")
    for _ in range(3):
        action = policy_agent.select_action(state, valid_actions)
        print(f"Policy agent selected: {action}")
        if action not in valid_actions:
            print("❌ CRITICAL: Policy agent selected invalid action!")
            return
    
    # Test 5: Check network predictions
    print("\n🔮 Test 5: Network Prediction Analysis")
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    if hasattr(policy_agent, 'network') and policy_agent.network:
        with torch.no_grad():
            policy_probs = policy_agent.network(state_tensor)
            print(f"Policy probabilities: {policy_probs.numpy()[0]}")
            print(f"Max prob: {policy_probs.max().item():.3f}, Min prob: {policy_probs.min().item():.3f}")
    else:
        print("❌ Policy agent has no network!")
    
    # Test 6: Quick training test
    print("\n🏋️ Test 6: Quick Training Test")
    print("Generating some training data...")
    from training.trainer import generate_minimax_training_data
    from agents.minimax_agent import MinimaxAgent
    
    minimax_agent = MinimaxAgent(depth=1)  # Very weak minimax
    training_data = generate_minimax_training_data(minimax_agent, num_games=5)
    print(f"Generated {len(training_data)} training examples")
    
    if len(training_data) > 0:
        # Save initial loss
        initial_loss = trainer.train_policy_network(training_data, epochs=1, return_loss=True)
        print(f"Initial policy loss: {initial_loss:.3f}")
        
        # Train a bit more
        final_loss = trainer.train_policy_network(training_data, epochs=10, return_loss=True)
        print(f"Final policy loss: {final_loss:.3f}")
        
        if final_loss < initial_loss:
            print("✅ Network is learning (loss decreased)")
        else:
            print("❌ Network not learning (loss increased or stayed same)")
    
    print("\n🔍 Debug complete!")

def check_training_data_quality():
    """Check if training data has good quality."""
    print("\n📚 Checking training data quality...")
    
    from training.trainer import generate_minimax_training_data
    from agents.minimax_agent import MinimaxAgent
    
    minimax_agent = MinimaxAgent(depth=2)
    training_data = generate_minimax_training_data(minimax_agent, num_games=10)
    
    if not training_data:
        print("❌ No training data generated!")
        return
    
    print(f"Generated {len(training_data)} training examples")
    
    # Check policy targets
    policies = [example[1] for example in training_data]
    policy_array = np.array(policies)
    
    print(f"Policy target stats:")
    print(f"  Shape: {policy_array.shape}")
    print(f"  Mean: {policy_array.mean():.3f}")
    print(f"  Std: {policy_array.std():.3f}")
    print(f"  Min: {policy_array.min():.3f}")
    print(f"  Max: {policy_array.max():.3f}")
    
    # Check if policies are reasonable (should sum to ~1.0)
    policy_sums = policy_array.sum(axis=1)
    print(f"Policy sums: mean={policy_sums.mean():.3f}, std={policy_sums.std():.3f}")
    
    # Check value targets
    values = [example[2] for example in training_data]
    value_array = np.array(values)
    
    print(f"Value target stats:")
    print(f"  Mean: {value_array.mean():.3f}")
    print(f"  Std: {value_array.std():.3f}")
    print(f"  Min: {value_array.min():.3f}")
    print(f"  Max: {value_array.max():.3f}")
    
    # Values should be between -1 and 1
    if value_array.min() < -1.1 or value_array.max() > 1.1:
        print("❌ Value targets out of expected range [-1, 1]!")
    else:
        print("✅ Value targets in expected range")

if __name__ == "__main__":
    debug_mcts_issue()
    check_training_data_quality()