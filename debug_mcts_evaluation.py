#!/usr/bin/env python3
"""
Debug MCTS evaluation to identify why win rates are decreasing.
"""

import numpy as np
from training.trainer import Trainer
from agents.policy_agent import PolicyAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from networks.policy_network import PolicyNetwork

def test_mcts_consistency():
    """Test if MCTS agent produces consistent results."""
    print("🔍 Testing MCTS agent consistency...")
    
    # Create MCTS agents with different simulation counts
    mcts_100 = MCTSAgent(simulations=100, name="MCTS-100")
    mcts_1000 = MCTSAgent(simulations=1000, name="MCTS-1000")
    
    # Create a simple policy agent
    policy_agent = PolicyAgent("TestPolicy")
    policy_agent.set_network(PolicyNetwork())
    
    # Create trainer
    trainer = Trainer(save_dir='./debug_models')
    
    print("\n1. Testing PolicyAgent vs different MCTS strengths:")
    
    # Test vs MCTS-100
    result_100 = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_100, num_games=10)
    print(f"   vs MCTS-100: {result_100['win_rate']:.1%}")
    
    # Test vs MCTS-1000  
    result_1000 = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_1000, num_games=10)
    print(f"   vs MCTS-1000: {result_1000['win_rate']:.1%}")
    
    print("\n2. Testing MCTS vs Random baseline:")
    
    random_agent = RandomAgent("Random")
    
    # MCTS-100 vs Random
    mcts100_vs_random = trainer.evaluate_agent_vs_baseline(mcts_100, random_agent, num_games=10)
    print(f"   MCTS-100 vs Random: {mcts100_vs_random['win_rate']:.1%}")
    
    # MCTS-1000 vs Random  
    mcts1000_vs_random = trainer.evaluate_agent_vs_baseline(mcts_1000, random_agent, num_games=10)
    print(f"   MCTS-1000 vs Random: {mcts1000_vs_random['win_rate']:.1%}")
    
    print("\n3. Testing player_id assignment consistency:")
    
    # Test multiple evaluations to see if results are consistent
    results = []
    for i in range(3):
        result = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_100, num_games=5)
        results.append(result['win_rate'])
        print(f"   Run {i+1}: {result['win_rate']:.1%}")
    
    win_rate_variance = np.var(results)
    print(f"   Variance in win rates: {win_rate_variance:.4f}")
    
    if win_rate_variance > 0.01:  # More than 1% variance is suspicious for same matchup
        print("   ⚠️  High variance detected - possible inconsistent player_id assignment")
    else:
        print("   ✅ Win rates are consistent")

def test_mcts_player_id_bug():
    """Test specific player_id assignment issue."""
    print("\n🔍 Testing MCTS player_id assignment...")
    
    # Create fresh agents
    policy_agent = PolicyAgent("TestPolicy")
    policy_agent.set_network(PolicyNetwork())
    
    mcts_agent = MCTSAgent(simulations=50, name="MCTS Test")
    
    # Manually test player_id assignment
    print(f"Initial MCTS player_id: {mcts_agent.player_id}")
    
    mcts_agent.set_player_id(1)
    print(f"After setting to 1: {mcts_agent.player_id}")
    
    mcts_agent.set_player_id(-1)
    print(f"After setting to -1: {mcts_agent.player_id}")
    
    # Test that MCTS uses player_id correctly
    from connect4_env import GymnasiumConnectFour
    env = GymnasiumConnectFour()
    env.reset()
    
    board = env.board.copy()
    action_mask = env._action_mask()
    
    # Test with player_id = 1
    mcts_agent.set_player_id(1)
    action1 = mcts_agent.select_action(board, action_mask)
    print(f"Action selected as player 1: {action1}")
    
    # Test with player_id = -1
    mcts_agent.set_player_id(-1)
    action2 = mcts_agent.select_action(board, action_mask)
    print(f"Action selected as player -1: {action2}")
    
    if action1 == action2:
        print("   ⚠️  Same action selected for both players - might indicate player_id not being used correctly")
    else:
        print("   ✅ Different actions for different player_ids")

if __name__ == "__main__":
    test_mcts_consistency()
    test_mcts_player_id_bug()