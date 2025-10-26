#!/usr/bin/env python3
"""
Test if the first-player advantage is environment bug or realistic.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent

def test_first_player_advantage():
    print("🔍 Testing First Player Advantage")
    
    # Test Random vs Random (should be roughly even regardless of first player)
    random1 = RandomAgent("Random1")
    random2 = RandomAgent("Random2")
    
    player1_wins = 0
    player2_wins = 0
    ties = 0
    
    for game in range(100):  # More games for better statistics
        env = GymnasiumConnectFour()
        env.reset()
        
        terminated = False
        move_count = 0
        max_moves = 42
        
        while not terminated and move_count < max_moves:
            # Get current player
            current_agent = random1 if env.current_player == 1 else random2
            
            # Get action
            try:
                # Try neural network interface first
                board = env.board.copy()
                action_mask = env._action_mask()
                action = current_agent.select_action(board, action_mask)
            except TypeError:
                # Fallback to environment interface
                action = current_agent.select_action(env)
            
            # Make move
            step_result = env.step(action)
            if len(step_result) == 6:
                _, reward, terminated, _, info, _ = step_result
            elif len(step_result) == 5:
                _, reward, terminated, _, info = step_result
            else:
                _, reward, terminated, info = step_result
            
            move_count += 1
            
            # Check for winner
            if terminated:
                if 'winner' in info:
                    winner = info['winner']
                elif reward != 0:
                    winner = env.current_player * -1  # Previous player won
                else:
                    winner = 0  # Tie
                
                if winner == 1:
                    player1_wins += 1
                elif winner == -1:
                    player2_wins += 1
                else:
                    ties += 1
                break
        
        if not terminated:
            ties += 1
    
    print(f"\nRandom vs Random (100 games):")
    print(f"Player 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Ties: {ties}")
    print(f"Player 1 win rate: {player1_wins/100:.1%}")
    
    if player1_wins > 70:
        print("❌ HUGE first-player advantage - likely environment bug!")
    elif player1_wins > 60:
        print("⚠️ Significant first-player advantage - investigate")
    elif player1_wins >= 45 and player1_wins <= 55:
        print("✅ Reasonable balance between players")
    else:
        print("🤔 Unexpected result - investigate")

if __name__ == "__main__":
    test_first_player_advantage()