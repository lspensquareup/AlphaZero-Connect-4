#!/usr/bin/env python3
"""
Test for inherent player bias in Connect-4 environment.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent
import numpy as np

def test_random_vs_random(num_games=100):
    """Test random vs random to see if there's a player bias."""
    player1_wins = 0
    player2_wins = 0
    ties = 0
    
    print(f"Testing {num_games} games of Random vs Random...")
    
    for game in range(num_games):
        env = GymnasiumConnectFour()
        env.reset()
        
        agent1 = RandomAgent("Random1")
        agent2 = RandomAgent("Random2")
        
        terminated = False
        max_moves = 42
        move_count = 0
        
        while not terminated and move_count < max_moves:
            current_agent = agent1 if env.current_player == 1 else agent2
            action = current_agent.select_action(env)
            
            step_result = env.step(action)
            if len(step_result) == 6:
                _, reward, terminated, _, info, _ = step_result
            elif len(step_result) == 5:
                _, reward, terminated, _, info = step_result
            else:
                _, reward, terminated, info = step_result
            
            move_count += 1
            
            if terminated:
                if 'winner' in info:
                    winner = info['winner']
                    if winner == 1:
                        player1_wins += 1
                        if game < 10:  # Show first 10 games for debugging
                            print(f"Game {game+1}: Player 1 wins after {move_count} moves")
                    elif winner == -1:
                        player2_wins += 1
                        if game < 10:
                            print(f"Game {game+1}: Player 2 wins after {move_count} moves")
                    else:
                        ties += 1
                        if game < 10:
                            print(f"Game {game+1}: Tie after {move_count} moves")
                else:
                    ties += 1
                    if game < 10:
                        print(f"Game {game+1}: Tie (no winner info) after {move_count} moves")
                break
        
        if not terminated:
            ties += 1
            if game < 10:
                print(f"Game {game+1}: Tie (max moves) after {move_count} moves")
    
    print(f"\nResults after {num_games} games:")
    print(f"Player 1 wins: {player1_wins} ({player1_wins/num_games*100:.1f}%)")
    print(f"Player 2 wins: {player2_wins} ({player2_wins/num_games*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_games*100:.1f}%)")
    
    # In Connect-4, Player 1 (who goes first) has a theoretical advantage
    # But it shouldn't be 100%!
    if player1_wins == num_games:
        print("\n🚨 CRITICAL BUG: Player 1 wins 100% of games!")
        return False
    elif player1_wins > num_games * 0.7:
        print(f"\n⚠️  Possible bias: Player 1 wins {player1_wins/num_games*100:.1f}% (expected ~52-58%)")
        return True
    else:
        print(f"\n✅ Player 1 advantage seems reasonable: {player1_wins/num_games*100:.1f}%")
        return True

if __name__ == "__main__":
    test_random_vs_random(50)