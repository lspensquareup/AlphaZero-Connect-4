#!/usr/bin/env python3
"""
Simple debug test to isolate the 50% evaluation issue.
"""

import torch
import numpy as np
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from agents.policy_agent import PolicyAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

def test_simple_evaluation():
    print("🔍 Simple Evaluation Debug Test")
    
    # Create a completely random policy agent (should lose to minimax)
    policy_net = PolicyNetwork(hidden_size=64)
    random_policy_agent = PolicyAgent(name="Random Policy")
    random_policy_agent.set_network(policy_net.eval())
    random_policy_agent.temperature = 0.1
    
    # Create agents to test against
    random_agent = RandomAgent(name="Random")
    minimax_agent = MinimaxAgent(depth=2)
    
    trainer = Trainer()
    
    print("\n1. Random Policy vs Random Agent:")
    results = trainer.evaluate_agent_vs_baseline(random_policy_agent, random_agent, num_games=20)
    print(f"Result: {results['win_rate']:.1%} (should be ~50%)")
    
    print("\n2. Random Policy vs Minimax-2:")
    results = trainer.evaluate_agent_vs_baseline(random_policy_agent, minimax_agent, num_games=20)
    print(f"Result: {results['win_rate']:.1%} (should be <50%, probably ~20-30%)")
    
    print("\n3. Random Agent vs Minimax-2:")
    results = trainer.evaluate_agent_vs_baseline(random_agent, minimax_agent, num_games=20)
    print(f"Result: {results['win_rate']:.1%} (should be <50%, probably ~20-30%)")
    
    print("\n4. Testing if minimax is deterministic:")
    # Test if minimax makes the same move in same position
    from connect4_env import GymnasiumConnectFour
    env = GymnasiumConnectFour()
    env.reset()
    
    # Make minimax choose the same position 5 times
    moves = []
    for i in range(5):
        action = minimax_agent.select_action(env)
        moves.append(action)
    
    print(f"Minimax moves for same position: {moves}")
    if len(set(moves)) == 1:
        print("✅ Minimax is deterministic")
    else:
        print("❌ Minimax is not deterministic - this is a problem!")
    
    print("\n5. Testing neural network randomness:")
    # Test if neural network gives different results
    board = env.board.copy()
    action_mask = env._action_mask()
    
    nn_moves = []
    for i in range(5):
        action = random_policy_agent.select_action(board, action_mask)
        nn_moves.append(action)
    
    print(f"Neural network moves for same position: {nn_moves}")
    if len(set(nn_moves)) > 1:
        print("✅ Neural network has randomness")
    else:
        print("❌ Neural network is deterministic with temp=0.1")

if __name__ == "__main__":
    test_simple_evaluation()