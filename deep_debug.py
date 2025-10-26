#!/usr/bin/env python3
"""
Deep debug of the evaluation system to find why it's exactly 50% everywhere.
"""

import torch
import numpy as np
from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from agents.policy_agent import PolicyAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

def debug_evaluation_system():
    print("🔍 Deep Debug: Why Exactly 50% Everywhere?")
    
    # Create a simple policy network
    policy_net = PolicyNetwork(hidden_size=64)
    trainer = Trainer()
    
    # Create agents
    policy_agent = PolicyAgent(name="Policy")
    policy_agent.set_network(policy_net.eval())
    policy_agent.temperature = 0.1
    
    minimax_d2 = MinimaxAgent(depth=2)
    minimax_d3 = MinimaxAgent(depth=3)
    random_agent = RandomAgent("Random")
    
    print("\n=== Test 1: Policy vs Different Minimax Depths ===")
    
    # Test against different minimax depths
    for depth in [2, 3]:
        minimax_agent = MinimaxAgent(depth=depth)
        print(f"\nTesting vs Minimax depth {depth}:")
        
        # Run multiple small evaluations to see if results vary
        for trial in range(3):
            result = trainer.evaluate_agent_vs_baseline(policy_agent, minimax_agent, num_games=4)
            print(f"  Trial {trial+1}: {result['wins']}W-{result['losses']}L-{result['ties']}T = {result['win_rate']:.1%}")
    
    print("\n=== Test 2: Check Individual Game Results ===")
    
    # Let's manually play a few games and see what happens
    wins_p1 = 0
    wins_p2 = 0
    ties = 0
    
    for game in range(10):
        print(f"\nGame {game+1}:")
        
        # Alternate who goes first
        if game % 2 == 0:
            player1, player2 = policy_agent, minimax_d2
            p1_name, p2_name = "Policy", "Minimax"
        else:
            player1, player2 = minimax_d2, policy_agent
            p1_name, p2_name = "Minimax", "Policy"
        
        print(f"  {p1_name} (P1) vs {p2_name} (P2)")
        
        # Play the game
        winner = trainer._play_single_evaluation_game(player1, player2)
        
        print(f"  Winner: {winner}")
        
        if winner == 1:
            wins_p1 += 1
            if p1_name == "Policy":
                print(f"  -> Policy wins as P1")
            else:
                print(f"  -> Minimax wins as P1")
        elif winner == -1:
            wins_p2 += 1
            if p2_name == "Policy":
                print(f"  -> Policy wins as P2")
            else:
                print(f"  -> Minimax wins as P2")
        else:
            ties += 1
            print(f"  -> Tie")
    
    print(f"\nManual game results:")
    print(f"Player 1 wins: {wins_p1}")
    print(f"Player 2 wins: {wins_p2}")
    print(f"Ties: {ties}")
    
    print("\n=== Test 3: Check Win Counting Logic ===")
    
    # Test the win counting logic directly
    print("Testing win counting with known outcomes...")
    
    # Simulate some game results
    test_results = [
        (1, "policy", "minimax"),   # Policy wins as player 1
        (-1, "minimax", "policy"),  # Minimax wins as player 2
        (1, "minimax", "policy"),   # Minimax wins as player 1
        (-1, "policy", "minimax"),  # Policy wins as player 2
        (0, "policy", "minimax"),   # Tie
    ]
    
    policy_wins = 0
    policy_losses = 0
    policy_ties = 0
    
    for winner, p1_type, p2_type in test_results:
        print(f"Winner: {winner}, P1: {p1_type}, P2: {p2_type}")
        
        # This is the logic from trainer.py
        if winner == 0:
            policy_ties += 1
            print("  -> Tie")
        elif (winner == 1 and p1_type == "policy") or (winner == -1 and p2_type == "policy"):
            policy_wins += 1
            print("  -> Policy wins")
        else:
            policy_losses += 1
            print("  -> Policy loses")
    
    print(f"\nCounting test results:")
    print(f"Policy: {policy_wins}W-{policy_losses}L-{policy_ties}T")
    print(f"Win rate: {policy_wins/(policy_wins+policy_losses+policy_ties):.1%}")

if __name__ == "__main__":
    debug_evaluation_system()