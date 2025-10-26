#!/usr/bin/env python3
"""
Deep debug of MCTS algorithm to find the decreasing win rate issue.
"""

import numpy as np
from agents.mcts_agent import MCTSAgent, MCTSNode
from agents.random_agent import RandomAgent
from training.trainer import Trainer

def test_mcts_rewards():
    """Test MCTS reward calculation and backpropagation."""
    print("🔍 Testing MCTS reward calculation...")
    
    # Create a simple board state
    board = np.zeros((6, 7), dtype=int)
    
    # Test reward calculation for different scenarios
    node = MCTSNode(board, player=1)
    
    print(f"Empty board - Player 1 reward: {node.get_reward(1)}")
    print(f"Empty board - Player -1 reward: {node.get_reward(-1)}")
    
    # Create a winning position for player 1
    winning_board = np.zeros((6, 7), dtype=int)
    winning_board[5, 0:4] = 1  # Four in a row for player 1
    
    winning_node = MCTSNode(winning_board, player=1)
    print(f"Player 1 winning - Player 1 reward: {winning_node.get_reward(1)}")
    print(f"Player 1 winning - Player -1 reward: {winning_node.get_reward(-1)}")
    
    # Create a winning position for player -1
    losing_board = np.zeros((6, 7), dtype=int)
    losing_board[5, 0:4] = -1  # Four in a row for player -1
    
    losing_node = MCTSNode(losing_board, player=-1)
    print(f"Player -1 winning - Player 1 reward: {losing_node.get_reward(1)}")
    print(f"Player -1 winning - Player -1 reward: {losing_node.get_reward(-1)}")

def test_mcts_simulation_quality():
    """Test the quality of MCTS simulations."""
    print("\n🔍 Testing MCTS simulation quality...")
    
    # Test against random agent multiple times with different sim counts
    random_agent = RandomAgent("Random")
    trainer = Trainer(save_dir='./debug_models')
    
    sim_counts = [50, 100, 200, 500]
    
    for sims in sim_counts:
        mcts_agent = MCTSAgent(simulations=sims, name=f"MCTS-{sims}")
        
        # Run multiple evaluations to get average
        win_rates = []
        for _ in range(3):
            result = trainer.evaluate_agent_vs_baseline(mcts_agent, random_agent, num_games=10)
            win_rates.append(result['win_rate'])
        
        avg_win_rate = np.mean(win_rates)
        std_win_rate = np.std(win_rates)
        
        print(f"MCTS-{sims}: {avg_win_rate:.1%} ± {std_win_rate:.1%}")

def test_rollout_bias():
    """Test if MCTS rollouts have player bias."""
    print("\n🔍 Testing MCTS rollout bias...")
    
    # Create a board where player 1 and -1 should have equal chances
    board = np.zeros((6, 7), dtype=int)
    
    # Test rollouts from player 1 perspective
    mcts_agent = MCTSAgent(simulations=10, name="MCTS Test")
    
    rewards_as_p1 = []
    rewards_as_p2 = []
    
    # Run many rollouts as player 1
    mcts_agent.set_player_id(1)
    for _ in range(100):
        node = MCTSNode(board, player=1)
        reward = mcts_agent._rollout(node, root_player=1)
        rewards_as_p1.append(reward)
    
    # Run many rollouts as player -1
    mcts_agent.set_player_id(-1)
    for _ in range(100):
        node = MCTSNode(board, player=-1)
        reward = mcts_agent._rollout(node, root_player=-1)
        rewards_as_p2.append(reward)
    
    avg_reward_p1 = np.mean(rewards_as_p1)
    avg_reward_p2 = np.mean(rewards_as_p2)
    
    print(f"Average reward as Player 1: {avg_reward_p1:.3f}")
    print(f"Average reward as Player -1: {avg_reward_p2:.3f}")
    
    if abs(avg_reward_p1 - avg_reward_p2) > 0.1:
        print("⚠️  Significant bias detected in rollouts!")
    else:
        print("✅ Rollouts appear unbiased")

def debug_specific_evaluation():
    """Debug the specific evaluation that's causing issues."""
    print("\n🔍 Debugging neural network vs MCTS evaluation...")
    
    from agents.policy_agent import PolicyAgent
    from networks.policy_network import PolicyNetwork
    
    # Create agents with same setup as dashboard
    policy_agent = PolicyAgent("PolicyAgent")
    policy_agent.set_network(PolicyNetwork())
    
    mcts_agent = MCTSAgent(simulations=100, name="MCTS Agent")  # Same as dashboard
    
    trainer = Trainer(save_dir='./debug_models')
    
    print("Running 5 separate evaluations to check for patterns...")
    
    for i in range(5):
        result = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_agent, num_games=10)
        print(f"Eval {i+1}: {result['win_rate']:.1%} ({result['wins']}W-{result['losses']}L-{result['ties']}T)")

if __name__ == "__main__":
    test_mcts_rewards()
    test_mcts_simulation_quality()
    test_rollout_bias()
    debug_specific_evaluation()