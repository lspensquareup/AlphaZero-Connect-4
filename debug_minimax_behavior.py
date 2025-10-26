#!/usr/bin/env python3
"""
Debug the Minimax agent's behavior as Player 1 vs Player 2.
"""

from connect4_env import GymnasiumConnectFour
from agents.minimax_agent import MinimaxAgent
import numpy as np

def debug_minimax_behavior():
    """Debug how Minimax behaves differently as Player 1 vs Player 2."""
    
    print("🔍 Testing Minimax behavior as different players")
    
    # Test 1: Minimax as Player 1
    print("\n=== Test 1: Minimax as Player 1 ===")
    env1 = GymnasiumConnectFour()
    env1.reset()
    
    minimax1 = MinimaxAgent(depth=2, name="Minimax-P1")
    
    print(f"Board state: \n{env1.board}")
    print(f"Current player: {env1.current_player}")
    print(f"Action mask: {env1._action_mask()}")
    
    action1 = minimax1.select_action(env1)
    print(f"Minimax selected action: {action1}")
    
    # Test 2: Minimax as Player 2 (simulate Player 1 making a move first)
    print("\n=== Test 2: Minimax as Player 2 ===")
    env2 = GymnasiumConnectFour()
    env2.reset()
    
    # Simulate Player 1 move
    env2.step(3)  # Player 1 plays in column 3
    print(f"After Player 1 move:")
    print(f"Board state: \n{env2.board}")
    print(f"Current player: {env2.current_player}")
    print(f"Action mask: {env2._action_mask()}")
    
    minimax2 = MinimaxAgent(depth=2, name="Minimax-P2")
    action2 = minimax2.select_action(env2)
    print(f"Minimax selected action: {action2}")
    
    # Test 3: Check if the Minimax agent is correctly understanding its player role
    print("\n=== Test 3: Check player perspective ===")
    
    # Manually check what the minimax agent sees
    print("Creating identical board positions for both players...")
    
    # Position where Player 1 is about to win
    test_env = GymnasiumConnectFour()
    test_env.reset()
    
    # Set up a position: Player 1 has 3 in a row, needs to block or win
    test_env.board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0]  # Player 1 has 3 in a row, column 3 would win
    ])
    
    print("Test board with Player 1 about to win:")
    print(test_env.board)
    
    # Test as Player 1 (should win by playing column 3)
    test_env.current_player = 1
    minimax_as_p1 = MinimaxAgent(depth=2, name="Test-P1")
    action_p1 = minimax_as_p1.select_action(test_env)
    print(f"Minimax as Player 1 chooses action: {action_p1} (should be 3 to win)")
    
    # Test as Player 2 (should block by playing column 3)
    test_env.current_player = -1
    minimax_as_p2 = MinimaxAgent(depth=2, name="Test-P2") 
    action_p2 = minimax_as_p2.select_action(test_env)
    print(f"Minimax as Player 2 chooses action: {action_p2} (should be 3 to block)")

if __name__ == "__main__":
    debug_minimax_behavior()