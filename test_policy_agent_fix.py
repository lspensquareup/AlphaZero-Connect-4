#!/usr/bin/env python3
"""
Test the fixed PolicyAgent to make sure it can select actions.
"""

from agents.policy_agent import PolicyAgent
from networks.policy_network import PolicyNetwork
from connect4_env import GymnasiumConnectFour
import numpy as np
import torch

def test_policy_agent():
    """Test PolicyAgent to verify it works correctly."""
    print("🔍 Testing PolicyAgent...")
    
    # Create environment
    env = GymnasiumConnectFour()
    env.reset()
    
    # Test 1: PolicyAgent without network (should fallback to random)
    print("\n=== Test 1: PolicyAgent without network ===")
    agent_no_network = PolicyAgent("PolicyAgent-NoNetwork")
    agent_no_network.set_player_id(1)
    
    try:
        action = agent_no_network.select_action(env.board, env._action_mask())
        print(f"✅ No network fallback: selected action {action}")
        print(f"Action type: {type(action)}, Valid action: {action in range(7)}")
    except Exception as e:
        print(f"❌ No network test failed: {e}")
        return False
    
    # Test 2: PolicyAgent with network
    print("\n=== Test 2: PolicyAgent with network ===")
    try:
        # Create a simple network
        network = PolicyNetwork()
        agent_with_network = PolicyAgent("PolicyAgent-WithNetwork")
        agent_with_network.set_network(network)
        agent_with_network.set_player_id(1)
        
        action = agent_with_network.select_action(env.board, env._action_mask())
        print(f"✅ With network: selected action {action}")
        print(f"Action type: {type(action)}, Valid action: {action in range(7)}")
        
        # Test action probabilities
        probs = agent_with_network.get_action_probabilities(env.board, env._action_mask())
        print(f"✅ Action probabilities: {probs}")
        print(f"Probabilities sum to: {probs.sum():.3f}")
        
    except Exception as e:
        print(f"❌ With network test failed: {e}")
        return False
    
    # Test 3: Try making actual moves
    print("\n=== Test 3: Playing a game ===")
    try:
        for move in range(5):
            action = agent_with_network.select_action(env.board, env._action_mask())
            step_result = env.step(action)
            print(f"Move {move+1}: Action {action}, Result: {len(step_result)} elements")
            
            if len(step_result) >= 3 and step_result[2]:  # terminated
                print("Game ended")
                break
    except Exception as e:
        print(f"❌ Game test failed: {e}")
        return False
    
    print("\n✅ All PolicyAgent tests passed!")
    return True

if __name__ == "__main__":
    test_policy_agent()