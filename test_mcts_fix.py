#!/usr/bin/env python3
"""
Quick test of MCTS agent to verify the fix.
"""

from agents.mcts_agent import MCTSAgent
from connect4_env import GymnasiumConnectFour
import numpy as np

def test_mcts_agent():
    """Test MCTS agent to make sure it doesn't crash."""
    print("🔍 Testing MCTS agent...")
    
    # Create environment and agent
    env = GymnasiumConnectFour()
    env.reset()
    
    mcts_agent = MCTSAgent(simulations=50, name="TestMCTS")
    mcts_agent.set_player_id(1)
    
    try:
        # Test action selection
        action_mask = env._action_mask()
        print(f"Board shape: {env.board.shape}")
        print(f"Action mask: {action_mask}")
        print(f"Action mask type: {type(action_mask)}")
        
        action = mcts_agent.select_action(env.board, action_mask)
        print(f"Selected action: {action}")
        print(f"Action type: {type(action)}")
        
        # Make the move
        step_result = env.step(action)
        print(f"Move successful: {len(step_result)} elements returned")
        
        print("✅ MCTS agent working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ MCTS agent error: {e}")
        return False

if __name__ == "__main__":
    test_mcts_agent()