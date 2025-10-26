#!/usr/bin/env python3
"""
Test the trainer's evaluation method directly to isolate the bug.
"""

import torch
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def test_trainer_method():
    print("🔍 Testing Trainer's Evaluation Method Directly")
    
    # Create agents
    trainer = Trainer()
    policy_net = PolicyNetwork(hidden_size=64)
    
    policy_agent = PolicyAgent(name="Policy")
    policy_agent.set_network(policy_net.eval())
    policy_agent.temperature = 0.1
    
    random_agent = RandomAgent("Random")
    minimax_agent = MinimaxAgent(depth=2)
    
    print("\n=== Test 1: Direct calls to _play_single_evaluation_game ===")
    
    # Test 10 individual games manually
    results = []
    for i in range(10):
        # Alternate who goes first
        if i % 2 == 0:
            player1, player2 = policy_agent, minimax_agent
            p1_name, p2_name = "Policy", "Minimax"
        else:
            player1, player2 = minimax_agent, policy_agent
            p1_name, p2_name = "Minimax", "Policy"
        
        winner = trainer._play_single_evaluation_game(player1, player2)
        results.append((winner, p1_name, p2_name))
        print(f"Game {i+1}: {p1_name} (P1) vs {p2_name} (P2) -> Winner: {winner}")
    
    # Count results
    policy_wins = 0
    minimax_wins = 0
    ties = 0
    
    for winner, p1_name, p2_name in results:
        if winner == 1:
            if p1_name == "Policy":
                policy_wins += 1
            else:
                minimax_wins += 1
        elif winner == -1:
            if p2_name == "Policy":
                policy_wins += 1
            else:
                minimax_wins += 1
        else:
            ties += 1
    
    print(f"\nManual count: Policy {policy_wins}, Minimax {minimax_wins}, Ties {ties}")
    
    print("\n=== Test 2: Using evaluate_agent_vs_baseline ===")
    
    # Now test using the full evaluation method
    result = trainer.evaluate_agent_vs_baseline(policy_agent, minimax_agent, num_games=10)
    print(f"Trainer method: {result['wins']}W-{result['losses']}L-{result['ties']}T = {result['win_rate']:.1%}")
    
    print("\n=== Test 3: Compare with Random vs Random ===")
    
    # Test random vs random for baseline
    random1 = RandomAgent("Random1")
    random2 = RandomAgent("Random2")
    
    result_random = trainer.evaluate_agent_vs_baseline(random1, random2, num_games=10)
    print(f"Random vs Random: {result_random['wins']}W-{result_random['losses']}L-{result_random['ties']}T = {result_random['win_rate']:.1%}")

if __name__ == "__main__":
    test_trainer_method()